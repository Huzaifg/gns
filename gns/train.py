import collections
import functools
import json
import numpy as np
import os
import torch
import pickle
import glob
import re

import tensorflow as tf
import tensorflow_datasets as tfds
import tree

from absl import flags
from absl import logging
from absl import app


from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 1, help='The batch size.')
flags.DEFINE_float('noise_std', 0, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 1  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.

  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])

  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Compute the number of particles per example.
  nparticles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = nparticles[tf.newaxis]

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position


def prepare_rollout_inputs(context, features):
  """Prepares an inputs trajectory for rollout."""
  out_dict = {**context}
  # Position is encoded as [sequence_length, nparticles, dim] but the model
  # expects [nparticles, sequence_length, dim].
  pos = tf.transpose(features['position'], [1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]
  # Remove the target from the input.
  out_dict['position'] = pos[:, :-1]
  # Compute the number of nodes
  out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
  if 'step_context' in features:
    out_dict['step_context'] = features['step_context']
  out_dict['is_trajectory'] = tf.constant([True], tf.bool)
  return out_dict, target_position


def batch_concat(dataset, batch_size):
  """We implement batching as concatenating on the leading axis."""

  # We create a dataset of datasets of length batch_size.
  windowed_ds = dataset.window(batch_size)

  # The plan is then to reduce every nested dataset by concatenating. We can
  # do this using tf.data.Dataset.reduce. This requires an initial state, and
  # then incrementally reduces by running through the dataset

  # Get initial state. In this case this will be empty tensors of the
  # correct shape.
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
          shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)

  # We run through the nest and concatenate each entry with the previous state.
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

  return windowed_ds.map(
      lambda *x: tree.map_structure(reduce_window, initial_state, x))


def prepare_input_data(
        data_path: str,
        batch_size: int = 2,
        mode: str = 'train',
        split: str = 'train'):
  """Prepares the input data for learning simulation from tfrecord.

  Args:
    data_path: the path to the dataset directory.
    batch_size: the number of graphs in a batch.
    mode: either 'train' or 'rollout'
    split: either 'train', 'valid' or 'test'.

  Returns:
    The input data for the learning simulation model.
  """
  # Loads the metadata of the dataset.
  metadata = reading_utils.read_metadata(data_path)
  # Set CPU as the only available physical device
  tf.config.set_visible_devices([], 'GPU')

  # Create a tf.data.Dataset from the TFRecord.
  split = "test"
  ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
  ds = ds.map(functools.partial(
      reading_utils.parse_serialized_simulation_example, metadata=metadata))

  if mode == 'rollout':
    ds = ds.map(prepare_rollout_inputs)
  elif mode == 'train':
    # Splits an entire trajectory into chunks of 7 steps.
    # Previous 5 velocities, current velocity and target.
    split_with_window = functools.partial(
        reading_utils.split_trajectory,
        window_length=INPUT_SEQUENCE_LENGTH + 1)
    ds = ds.flat_map(split_with_window)
    # Splits a chunk into input steps and target steps
    ds = ds.map(prepare_inputs)
    # If in train mode, repeat dataset forever and shuffle.
    # ds = ds.repeat()
    # ds = ds.shuffle(512)
    # Custom batching on the leading axis.
    ds = batch_concat(ds, batch_size)

  # Convert to numpy
  ds = tfds.as_numpy(ds)

  return ds


def rollout(
        simulator: learned_simulator.LearnedSimulator,
        features: torch.tensor,
        nsteps: int):
  """Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
  """
  initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in range(nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=features['n_particles_per_example'],
        particle_types=features['particle_type'],
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (features['particle_type'] ==
                      3).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 2)
    next_position = torch.where(
        kinematic_mask, next_position_ground_truth, next_position)
    predictions.append(next_position)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)

  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': features['particle_type'].cpu().numpy(),
  }

  return output_dict, loss


def predict(
        simulator: learned_simulator.LearnedSimulator,
        metadata: json,
        device):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.
    metadata: Metadata for test set.

  """

  # Load simulator
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    train(simulator)

  # Output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if FLAGS.mode == 'rollout' else 'valid'
  ds = prepare_input_data(FLAGS.data_path,
                          batch_size=FLAGS.batch_size,
                          mode='rollout', split=split)

  # Move model to device
  simulator.to(device)

  eval_loss = []
  with torch.no_grad():
    for example_i, (features, labels) in enumerate(ds):
      features['position'] = torch.tensor(
          features['position']).to(device)
      features['n_particles_per_example'] = torch.tensor(
          features['n_particles_per_example']).to(device)
      features['particle_type'] = torch.tensor(
          features['particle_type']).to(device)
      labels = torch.tensor(labels).to(device)

      nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      # Predict example rollout
      example_rollout, loss = rollout(simulator, features, nsteps)

      example_rollout['metadata'] = metadata
      print("Predicting example {} loss: {}".format(example_i, loss.mean()))
      eval_loss.append(torch.flatten(loss))
      
      # Save rollout in testing
      if FLAGS.mode == 'rollout':
        example_rollout['metadata'] = metadata
        filename = f'rollout_{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))

def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def train(
        simulator: learned_simulator.LearnedSimulator,
        device):
  """Train the model.

  Args:
    simulator: Get LearnedSimulator.
  """
  optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init)
  step = 0
  # If model_path does not exist create new directory and begin training.
  model_path = FLAGS.model_path
  if not os.path.exists(model_path):
    os.makedirs(model_path)
    

  # If model_path does exist and model_file and train_state_file exist continue training.
  if FLAGS.model_file is not None:

    if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f"{model_path}*model*pt")
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      FLAGS.model_file = f"model-{max_model_number}.pt"
      FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

    if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
      # load model
      simulator.load(model_path + FLAGS.model_file)

      # load train state
      train_state = torch.load(model_path + FLAGS.train_state_file)
      # set optimizer state
      optimizer = torch.optim.Adam(simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device)
      # set global train state
      step = train_state["global_train_state"].pop("step")
 
    else:
      msg = f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found."
      raise FileNotFoundError(msg) 

  simulator.train()
  simulator.to(device)

  ds = prepare_input_data(FLAGS.data_path,
                          batch_size=FLAGS.batch_size)

  print(f"device = {device}")

  # tfrecord can contain one or more simulations
  # these simulations can contain different numbers of particles
  # to save these in arrays we need to only group those with identical number of particles
  suite_particle_types = []
  suite_positions = []
  suite_n_particle_per_example = []
  suite_labels = []

  try:

    # each grouping of simulations with the same number of particles will be saved as a set
    # these sets will then be concatenated and appended to the suites above.
    set_particle_type = []
    set_position = []
    set_n_particles_per_example = []
    set_labels = []

    previous_num_particles = None
    for features, labels in ds:
      particle_type = features["particle_type"]
      position = features['position']
      n_particles_per_example = features["n_particles_per_example"]
      # labels = labels

      if previous_num_particles is None:
        previous_num_particles = n_particles_per_example

      if previous_num_particles != n_particles_per_example:
        # must be starting a new set
        # append old set to suite
        suite_particle_types.append(np.concatenate(set_particle_type))
        suite_positions.append(np.concatenate(set_position))
        suite_n_particle_per_example.append(np.concatenate(set_n_particles_per_example))
        suite_labels.append(np.concatenate(set_labels))

        # then reset the set to begin the next one.
        set_particle_type = []
        set_position = []
        set_n_particles_per_example = []
        set_labels = []

        # reset number of particles for new set
        previous_num_particles = n_particles_per_example

      # always append the latest data to the current set.
      set_particle_type.append(np.array([particle_type]))
      set_position.append(np.array([position]))
      set_n_particles_per_example.append(np.array([n_particles_per_example]))
      set_labels.append(np.array([labels]))

      # # Sample the noise to add to the inputs to the model during training.
      # sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
      #     features['position'], noise_std_last_step=FLAGS.noise_std).to(device)
      # non_kinematic_mask = (
      #     features['particle_type'] != 3).clone().detach().to(device)
      # sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

      # # Get the predictions and target accelerations.
      # pred_acc, target_acc = simulator.predict_accelerations(
      #     next_positions=labels.to(device),
      #     position_sequence_noise=sampled_noise.to(device),
      #     position_sequence=features['position'].to(device),
      #     nparticles_per_example=features['n_particles_per_example'].to(device),
      #     particle_types=features['particle_type'].to(device))

      # # Calculate the loss and mask out loss on kinematic particles
      # loss = (pred_acc - target_acc) ** 2
      # loss = loss.sum(dim=-1)
      # num_non_kinematic = non_kinematic_mask.sum()
      # loss = torch.where(non_kinematic_mask.bool(),
      #                    loss, torch.zeros_like(loss))
      # loss = loss.sum() / num_non_kinematic

      # # Computes the gradient of loss
      # optimizer.zero_grad()
      # loss.backward()
      # optimizer.step()

      # # Update learning rate
      # lr_new = FLAGS.lr_init * (FLAGS.lr_decay ** (step/FLAGS.lr_decay_steps))
      # for param in optimizer.param_groups:
      #   param['lr'] = lr_new

      # print('Training step: {}/{}. Loss: {}.'.format(step,
      #                                                FLAGS.ntraining_steps,
      #                                                loss))
      # # Save model state
      # if step % FLAGS.nsave_steps == 0:
      #   simulator.save(model_path + 'model-'+str(step)+'.pt')
      #   train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
      #   torch.save(train_state, f"{model_path}train_state-{step}.pt")

      # # Complete training
      # if (step >= FLAGS.ntraining_steps):
      #   break

      # step += 1

  except KeyboardInterrupt:
    pass

  # print(len(series_particle_type), series_particle_type[0].shape)
  # print(len(series_position), series_position[0].shape)
  # print(len(series_n_particles_per_example), series_n_particles_per_example[0].shape)
  # print(len(series_labels), series_labels[0].shape)

  # print(series_labels[0], series_labels[1], series_labels[2])

  # simulator.save(model_path + 'model-'+str(step)+'.pt')
  # train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
  # torch.save(train_state, f"{model_path}train_state-{step}.pt")

  # series_particle_type = np.concatenate(series_particle_type)
  # series_position = np.concatenate(series_position)
  # series_n_particles_per_example = np.concatenate(series_n_particles_per_example)
  # series_labels = np.concatenate(series_labels)

  suite_particle_types.append(np.concatenate(set_particle_type))
  suite_positions.append(np.concatenate(set_position))
  suite_n_particle_per_example.append(np.concatenate(set_n_particles_per_example))
  suite_labels.append(np.concatenate(set_labels))

  split = "test"
  np.savez_compressed(f"{split}.npz",
                      **{f"particle_type_{key}" : value for key, value in enumerate(suite_particle_types)},
                      **{f"position_{key}" : value for key, value in enumerate(suite_positions)},
                      **{f"n_particles_per_example_{key}" : value for key, value in enumerate(suite_n_particle_per_example)},
                      **{f"label_{key}" : value for key, value in enumerate(suite_labels)},
  )
  # np.save("train_ptype.npy", series_particle_type)
  # np.save("train_position.npy", series_position)
  # np.save("train_nparticles.npy", series_n_particles_per_example)
  # np.save("train_labels.npy", series_labels)

def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
  """

  # Normalization stats
  normalization_stats = {
      'acceleration': {
          'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                            acc_noise_std**2).to(device),
      },
      'velocity': {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device),
      },
  }

  simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=30,
      nedge_in=3,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      device=device)

  return simulator


def main(_):
  """Train or evaluates the model.

  """
  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
    device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')

  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path)
  simulator = _get_simulator(
      metadata, FLAGS.noise_std, FLAGS.noise_std, device)
  if FLAGS.mode == 'train':
    train(simulator, device)
  elif FLAGS.mode in ['valid', 'rollout']:
    predict(simulator, metadata, device)


if __name__ == '__main__':
  app.run(main)
