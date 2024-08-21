import collections
import json
import os
import pickle
import glob
import re
import sys
import wandb
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import particle_data_loader as pdl
from gns import distribute
from gns.args import Config
from gns.Dtype import DType

Stats = collections.namedtuple("Stats", ["mean", "std"])


def rollout(
    simulator: learned_simulator.LearnedSimulator,
    cfg: DictConfig,
    position: torch.tensor,
    particle_types: torch.tensor,
    material_property: torch.tensor,
    n_particles_per_example: torch.tensor,
    nsteps: int,
    device: torch.device,
):
    """
    Rolls out a trajectory by applying the model in sequence.

    Args:
      simulator: Learned simulator.
      position: Positions of particles (timesteps, nparticles, ndims)
      particle_types: Particles types with shape (nparticles)
      material_property: Friction angle normalized by tan() with shape (nparticles)
      n_particles_per_example
      nsteps: Number of steps.
      device: torch device.
    """

    if cfg.rollout.dtype_mode == 'half':
        dtype = torch.float16
    else:
        dtype = torch.float32

    initial_positions = position[:, : cfg.data.input_sequence_length]
    ground_truth_positions = position[:, cfg.data.input_sequence_length :]

    # This might need to be removed, do not remeber why I added it
    if(nsteps != ground_truth_positions.shape[1]):
        nsteps = ground_truth_positions.shape[1]

    current_positions = initial_positions
    predictions = torch.zeros(size = (nsteps, current_positions.shape[0], current_positions.shape[2]), device=device, dtype=dtype)
    print(f"Using datatype: {dtype}", flush=True)
    print(f"steps: {nsteps}")

    for step in tqdm(range(nsteps), total=nsteps):
        # Get next position with shape (nnodes, dim)
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
            material_property=material_property,
        )

        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (
            (particle_types == cfg.data.kinematic_particle_id)
            .clone()
            .detach()
            .to(device)
        )
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(
            -1, current_positions.shape[-1]
        )
        next_position = torch.where(
            kinematic_mask, next_position_ground_truth, next_position
        )
        predictions[step] = next_position

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1
        )

    # Predictions with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

    loss = (predictions - ground_truth_positions) ** 2

    output_dict = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": particle_types.cpu().numpy(),
        "material_property": (
            material_property.cpu().numpy() if material_property is not None else None
        ),
    }
    simulator.reset_graph_state() # This has to be called if more than 1 rollout example is processed (which could very well happen)
    return output_dict, loss


def predict(device: str, cfg: DictConfig):
    """Predict rollouts.

    Args:
      device: 'cpu' or 'cuda'.
      cfg: configuration dictionary.

    """
    if cfg.rollout.dtype_mode == 'half':
        dtype = DType.HALF
    elif cfg.rollout.dtype_mode == 'single':
        dtype = DType.SINGLE
    else:
        raise ValueError("Wrong input: Mixed precision inference does not make any sense")

    if cfg.rollout.graph_build_freq > 1:
        lazy_graph_update = True
    # Read metadata
    metadata = reading_utils.read_metadata(cfg.data.path, "rollout")
    simulator = _get_simulator(
        metadata,
        cfg.data.num_particle_types,
        cfg.data.noise_std,
        cfg.data.noise_std,
        device,
        dtype = dtype,
        lazy_graph_update = lazy_graph_update,
        graph_build_freq = cfg.rollout.graph_build_freq
    )

    # Load simulator
    if os.path.exists(cfg.model.path + cfg.model.file):
        simulator.load(cfg.model.path + cfg.model.file)
    else:
        raise Exception(f"Model does not exist at {cfg.model.path + cfg.model.file}")

    if dtype == DType.HALF:
        simulator.half()
    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(cfg.output.path):
        os.makedirs(cfg.output.path)

    # Use `valid`` set for eval mode if not use `test`
    split = (
        "test"
        if (cfg.mode == "rollout" or (not os.path.isfile("{cfg.data.path}valid.npz")))
        else "valid"
    )

    # Get dataset
    ds = pdl.get_data_loader(file_path=f"{cfg.data.path}{split}.npz", mode="trajectory", dtype=dtype)
    # See if our dataset has material property as feature
    test_dataset = pdl.ParticleDataset(f"{cfg.data.path}{split}.npz")
    n_features = test_dataset.get_num_features()
    if n_features == 3:  # `ds` has (positions, particle_type, material_property)
        material_property_as_feature = True
    elif n_features == 2:  # `ds` only has (positions, particle_type)
        material_property_as_feature = False
    else:
        raise NotImplementedError

    eval_loss = []
    with torch.no_grad():
        for example_i, features in enumerate(ds):
            print(f"processing example number {example_i}")
            positions = features[0].to(device)
            if metadata["sequence_length"] is not None:
                # If `sequence_length` is predefined in metadata,
                nsteps = metadata["sequence_length"] - cfg.data.input_sequence_length
            else:
                # If no predefined `sequence_length`, then get the sequence length
                sequence_length = positions.shape[1]
                nsteps = sequence_length - cfg.data.input_sequence_length
            particle_type = features[1].to(device)
            if material_property_as_feature:
                material_property = features[2].to(device)
                n_particles_per_example = torch.tensor(
                    [int(features[3])], dtype=torch.int32
                ).to(device)
            else:
                material_property = None
                n_particles_per_example = torch.tensor(
                    [int(features[2])], dtype=torch.int32
                ).to(device)

            # Predict example rollout
            example_rollout, loss = rollout(
                simulator,
                cfg,
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
                nsteps,
                device,
            )

            example_rollout["metadata"] = metadata
            print("Predicting example {} loss: {}".format(example_i, loss.mean()))
            eval_loss.append(torch.flatten(loss))

            # Save rollout in testing
            if cfg.mode == "rollout":
                example_rollout["metadata"] = metadata
                example_rollout["loss"] = loss.mean()
                filename = f"{cfg.output.filename}_ex{example_i}.pkl"
                filename = os.path.join(cfg.output.path, filename)
                with open(filename, "wb") as f:
                    pickle.dump(example_rollout, f)

    print(
        "Mean loss on rollout prediction: {}".format(torch.mean(torch.cat(eval_loss)))
    )


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


def acceleration_loss(pred_acc, target_acc, non_kinematic_mask):
    """
    Compute the loss between predicted and target accelerations.

    Args:
      pred_acc: Predicted accelerations.
      target_acc: Target accelerations.
      non_kinematic_mask: Mask for kinematic particles.
    """
    loss = (pred_acc - target_acc) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()
    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    return loss


def save_model_and_train_state(
    verbose,
    device,
    simulator,
    cfg,
    step,
    epoch,
    optimizer,
    train_loss,
    valid_loss,
    train_loss_hist,
    valid_loss_hist,
    use_dist,
):
    """Save model state

    Args:
      rank: local rank
      device: torch device type
      simulator: Trained simulator if not will undergo training.
      cfg: Configuration dictionary.
      step: step
      epoch: epoch
      optimizer: optimizer
      train_loss: training loss at current step
      valid_loss: validation loss at current step
      train_loss_hist: training loss history at each epoch
      valid_loss_hist: validation loss history at each epoch
    """
    if verbose:
        if not use_dist:
            simulator.save(cfg.model.path + "model-" + str(step) + ".pt")
        else:
            simulator.module.save(cfg.model.path + "model-" + str(step) + ".pt")

        train_state = dict(
            optimizer_state=optimizer.state_dict(),
            global_train_state={
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            loss_history={"train": train_loss_hist, "valid": valid_loss_hist},
        )
        torch.save(train_state, f"{cfg.model.path}train_state-{step}.pt")


def setup_simulator_and_optimizer(cfg, metadata, rank, world_size, device, use_dist, dtype=torch.float32):
    """Setup simulator and optimizer.

    Args:
        cfg: Configuration dictionary.
        metadata: Metadata.
        rank: Local rank.
        world_size: Total number of ranks.
        device: torch device type.
    """
    if device == torch.device("cuda"):
        serial_simulator = _get_simulator(
            metadata,
            cfg.data.num_particle_types,
            cfg.data.noise_std,
            cfg.data.noise_std,
            rank,
            dtype=dtype
        )
        if use_dist:
            simulator = DDP(serial_simulator.to("cuda"), device_ids=[rank])
        else:
            simulator = serial_simulator.to("cuda")
        optimizer = torch.optim.Adam(
            simulator.parameters(), lr=cfg.training.learning_rate.initial * world_size
        )
    else:
        simulator = _get_simulator(
            metadata,
            cfg.data.num_particle_types,
            cfg.data.noise_std,
            cfg.data.noise_std,
            device,
            dtype=float
        )
        optimizer = torch.optim.Adam(
            simulator.parameters(), lr=cfg.training.learning_rate.initial * world_size
        )
    return simulator, optimizer


def initialize_training(cfg, rank, world_size, device, use_dist, dtype=torch.float32):
    """Initialize training.

    Args:
      cfg: Configuration dictionary.
      rank: Local rank.
      world_size: Total number of ranks.
      device: torch device type.
      use_dist: use torch.distribute
    """
    metadata = reading_utils.read_metadata(cfg.data.path, "train")
    simulator, optimizer = setup_simulator_and_optimizer(
        cfg, metadata, rank, world_size, device, use_dist, dtype=dtype
    )
    return simulator, optimizer, metadata


def load_datasets(cfg, use_dist):
    # Train data loader
    train_dl = pdl.get_data_loader(
        file_path=f"{cfg.data.path}train.npz",
        mode="sample",
        input_sequence_length=cfg.data.input_sequence_length,
        batch_size=cfg.data.batch_size,
        use_dist=use_dist,
    )
    train_dataset = pdl.ParticleDataset(f"{cfg.data.path}train.npz")
    n_features = train_dataset.get_num_features()

    # Validation data loader
    valid_dl = None
    if cfg.training.validation_interval is not None:
        valid_dl = pdl.get_data_loader(
            file_path=f"{cfg.data.path}valid.npz",
            mode="sample",
            input_sequence_length=cfg.data.input_sequence_length,
            batch_size=cfg.data.batch_size,
            use_dist=use_dist,
        )
        valid_dataset = pdl.ParticleDataset(f"{cfg.data.path}valid.npz")
        if valid_dataset.get_num_features() != n_features:
            raise ValueError(
                f"`n_features` of `valid.npz` and `train.npz` should be the same"
            )

    return train_dl, valid_dl, n_features


def setup_wandb(cfg, metadata):
    """Setup wandb logging.

    Args:
        cfg: Configuration dictionary.
        metadata: Metadata.
    """
    wandb.init(
        project="GNS",
        config={
            "lr_init": cfg.training.learning_rate.initial,
            "lr_decay": cfg.training.learning_rate.decay,
            "lr_decay_steps": cfg.training.learning_rate.decay_steps,
            "batch_size": cfg.data.batch_size,
            "noise_std": cfg.data.noise_std,
            "ntraining_steps": cfg.training.steps,
            "dtype_mode": cfg.training.dtype_mode,
        }
    )
    # TODO: Make sure to read this sweep config in the training portion and use it
    config = wandb.config # Sweep config setup for later

    # Log metadata as text
    metadata_json = json.dumps(metadata, indent=4)
    wandb.log({"metadata": metadata_json})

    # Log configuration as text
    yaml_config = OmegaConf.to_yaml(cfg)
    wandb.log({"Config": yaml_config})

    # Log initial metrics
    initial_metrics = {"train_loss": 0, "valid_loss": 0}
    wandb.log(initial_metrics)
    return config

def prepare_data(example, device_id):
    """Prepare data for training or validation."""
    position = example[0][0].to(device_id)
    particle_type = example[0][1].to(device_id)

    if len(example[0]) == 4:  # if data loader includes material_property
        material_property = example[0][2].to(device_id)
        n_particles_per_example = example[0][3].to(device_id)
    elif len(example[0]) == 3:
        material_property = None
        n_particles_per_example = example[0][2].to(device_id)
    else:
        raise ValueError("Unexpected number of elements in the data loader")

    labels = example[1].to(device_id)

    return position, particle_type, material_property, n_particles_per_example, labels


def train(rank, cfg, world_size, device, verbose, use_dist):
    """Train the model.

    Args:
      rank: local rank
      cfg: configuration dictionary
      world_size: total number of ranks
      device: torch device type
      verbose: global rank 0 or cpu
      use_dist: use torch.distribute
    """
    if cfg.training.dtype_mode == 'half':
        raise ValueError("Wrong input: Half precision training is not recommeneded")
    elif cfg.training.dtype_mode == 'single':
        dtype = DType.SINGLE
    else:
        dtype = DType.MIXED
        do_autocast = True
    
    device_id = rank if device == torch.device("cuda") else device

    # Initialize simulator and optimizer
    simulator, optimizer, metadata = initialize_training(
        cfg, rank, world_size, device, use_dist, dtype=dtype
    )

    # Get simulator and optimizer
    scaler = GradScaler(enabled = do_autocast) #Grad scaler to prevent gradient vanishing with amp

    # Initialize training state
    step = 0
    epoch = 0
    steps_per_epoch = 0

    valid_loss = None
    train_loss = 0
    epoch_valid_loss = None

    train_loss_hist = []
    valid_loss_hist = []

    # If model_path does exist and model_file and train_state_file exist continue training.
    if cfg.model.file is not None and cfg.training.resume:
        if cfg.model.file == "latest" and cfg.model.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f"{cfg.model.path}*model*pt")
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            cfg.model.file = f"model-{max_model_number}.pt"
            cfg.model.train_state_file = f"train_state-{max_model_number}.pt"

        if os.path.exists(cfg.model.path + cfg.model.file) and os.path.exists(
            cfg.model.path + cfg.model.train_state_file
        ):
            # load model
            if use_dist:
                simulator.module.load(cfg.model.path + cfg.model.file)
            else:
                simulator.load(cfg.model.path + cfg.model.file)

            # load train state
            train_state = torch.load(cfg.model.path + cfg.model.train_state_file)

            # set optimizer state
            optimizer = torch.optim.Adam(
                simulator.module.parameters() if use_dist else simulator.parameters()
            )
            optimizer.load_state_dict(train_state["optimizer_state"])
            if "scaler" in train_state:
                scaler.load_state_dict(train_state["scaler"])
            optimizer_to(optimizer, device_id)

            # set global train state
            step = train_state["global_train_state"]["step"]
            epoch = train_state["global_train_state"]["epoch"]
            train_loss_hist = train_state["loss_history"]["train"]
            valid_loss_hist = train_state["loss_history"]["valid"]

        else:
            msg = f"Specified model_file {cfg.model.path + cfg.model.file} and train_state_file {cfg.model.path + cfg.model.train_state_file} not found."
            raise FileNotFoundError(msg)

    simulator.train()
    simulator.to(device_id)

    # Load datasets - Here no data type is passed because Automatic Mixed Precision will take care of it
    train_dl, valid_dl, n_features = load_datasets(cfg, use_dist)

    print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

    wandb_config = setup_wandb(cfg, metadata) if verbose else None

    try:
        num_epochs = max(1, (cfg.training.steps + len(train_dl) - 1) // len(train_dl))
        if verbose:
            print(f"Total epochs = {num_epochs}")
        for epoch in tqdm(
            range(epoch, num_epochs), desc="Training", unit="epoch", disable=not verbose
        ):
            if use_dist:
                torch.distributed.barrier()

            epoch_loss = 0.0
            steps_this_epoch = 0

            # Create a tqdm progress bar for each epoch
            with tqdm(
                # resume from one step after the checkpoint
                range(step % len(train_dl) + 1, len(train_dl)),
                desc=f"Epoch {epoch}",
                unit="batch",
                disable=not verbose,
            ) as pbar:
                for example in train_dl:
                    steps_per_epoch += 1
                    # Prepare data
                    (
                        position,
                        particle_type,
                        material_property,
                        n_particles_per_example,
                        labels,
                    ) = prepare_data(example, device_id)

                    n_particles_per_example = n_particles_per_example.to(device_id)
                    labels = labels.to(device_id)

                    sampled_noise = (
                        noise_utils.get_random_walk_noise_for_position_sequence(
                            position, noise_std_last_step=cfg.data.noise_std
                        ).to(device_id)
                    )
                    non_kinematic_mask = (
                        (particle_type != cfg.data.kinematic_particle_id)
                        .clone()
                        .detach()
                        .to(device_id)
                    )
                    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                    device_or_rank = rank if device == torch.device("cuda") else device
                    predict_fn = (
                        simulator.module.predict_accelerations
                        if use_dist
                        else simulator.predict_accelerations
                    )
                    with autocast(dtype = torch.float16, enabled = do_autocast):
                        pred_acc, target_acc = predict_fn(
                            next_positions=labels.to(device_or_rank),
                            position_sequence_noise=sampled_noise.to(device_or_rank),
                            position_sequence=position.to(device_or_rank),
                            nparticles_per_example=n_particles_per_example.to(
                                device_or_rank
                            ),
                            particle_types=particle_type.to(device_or_rank),
                            material_property=(
                                material_property.to(device_or_rank)
                                if n_features == 3
                                else None
                            ),
                        )

                        if (
                            cfg.training.validation_interval is not None
                            and step > 0
                            and step % cfg.training.validation_interval == 0
                        ):
                            if verbose:
                                sampled_valid_example = next(iter(valid_dl))
                                valid_loss = validation(
                                    simulator,
                                    sampled_valid_example,
                                    n_features,
                                    cfg,
                                    rank,
                                    device_id,
                                    do_autocast=do_autocast
                                )
                            wandb.log({"Loss/valid": valid_loss.item()}, step=step)

                        loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

                    train_loss = loss.item()
                    epoch_loss += train_loss
                    steps_this_epoch += 1

                    optimizer.zero_grad()
                    scaler.scale(loss).backward() # calculate scaled gradients
                    scaler.step(optimizer)
                    scaler.update()

                    lr_new = (
                        cfg.training.learning_rate.initial
                        * (
                            cfg.training.learning_rate.decay
                            ** (step / cfg.training.learning_rate.decay_steps)
                        )
                        * world_size
                    )
                    for param in optimizer.param_groups:
                        param["lr"] = lr_new

                    # Log training loss
                    if verbose:
                        wandb.log({"Loss/train": train_loss}, step=step)
                        wandb.log({"Learning Rate": lr_new}, step=step)

                    avg_loss = epoch_loss / steps_this_epoch
                    pbar.set_postfix(
                        loss=f"{train_loss:.2f}",
                        avg_loss=f"{avg_loss:.2f}",
                        lr=f"{lr_new:.2e}",
                    )
                    pbar.update(1)

                    if verbose and step % cfg.training.save_steps == 0:
                        save_model_and_train_state(
                            verbose,
                            device,
                            simulator,
                            cfg,
                            step,
                            epoch,
                            optimizer,
                            train_loss,
                            valid_loss,
                            train_loss_hist,
                            valid_loss_hist,
                            use_dist,
                        )

                    step += 1
                    if step >= cfg.training.steps:
                        break

            # Epoch level statistics
            avg_loss = torch.tensor([epoch_loss / steps_this_epoch]).to(device_id)
            if use_dist:
                torch.distributed.reduce(
                    avg_loss, dst=0, op=torch.distributed.ReduceOp.SUM
                )
                avg_loss /= world_size

            train_loss_hist.append((epoch, avg_loss.item()))

            if cfg.training.validation_interval is not None:
                sampled_valid_example = next(iter(valid_dl))
                epoch_valid_loss = validation(
                    simulator, sampled_valid_example, n_features, cfg, rank, device_id, do_autocast=do_autocast
                )
                if device == torch.device("cuda"):
                    torch.distributed.reduce(
                        epoch_valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM
                    )
                    epoch_valid_loss /= world_size
                valid_loss_hist.append((epoch, epoch_valid_loss.item()))

            if verbose:
                wandb.log({"Loss/train_epoch": avg_loss.item()}, step=step)
                if cfg.training.validation_interval is not None:
                    wandb.log({"Loss/valid_epoch": epoch_valid_loss.item()}, step=step)

            if step >= cfg.training.steps:
                break
    except KeyboardInterrupt:
        pass

    # Save model state on keyboard interrupt
    save_model_and_train_state(
        verbose,
        device,
        simulator,
        cfg,
        step,
        epoch,
        optimizer,
        train_loss,
        valid_loss,
        train_loss_hist,
        valid_loss_hist,
        use_dist,
    )

    if use_dist:
        distribute.cleanup()


def _get_simulator(
    metadata: json,
    num_particle_types: int,
    acc_noise_std: float,
    vel_noise_std: float,
    device: torch.device,
    dtype: DType = DType.SINGLE,
    lazy_graph_update: bool = False,
    graph_build_freq: int = 1
) -> learned_simulator.LearnedSimulator:
    """Instantiates the simulator.

    Args:
      metadata: JSON object with metadata.
      acc_noise_std: Acceleration noise std deviation.
      vel_noise_std: Velocity noise std deviation.
      device: PyTorch device 'cpu' or 'cuda'.
    """

    # Normalization stats
    normalization_stats = {
        "acceleration": {
            "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["acc_std"]) ** 2 + acc_noise_std**2
            ).to(device),
        },
        "velocity": {
            "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
            "std": torch.sqrt(
                torch.FloatTensor(metadata["vel_std"]) ** 2 + vel_noise_std**2
            ).to(device),
        },
    }

    # Get necessary parameters for loading simulator.
    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata["nnode_in"]
        nedge_in = metadata["nedge_in"]
    else:
        # Given that there is no additional node feature (e.g., material_property) except for:
        # (position (dim), velocity (dim*6), particle_type (16)),
        nnode_in = 37 if metadata["dim"] == 3 else 30
        nedge_in = metadata["dim"] + 1

    # Init simulator.
    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata["dim"],
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata["default_connectivity_radius"],
        boundaries=np.array(metadata["bounds"]),
        normalization_stats=normalization_stats,
        nparticle_types=num_particle_types,
        particle_type_embedding_size=16,
        boundary_clamp_limit=(
            metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0
        ),
        device=device,
        dtype=dtype,
        lazy_graph_update=lazy_graph_update,
        graph_build_freq=graph_build_freq
    )

    return simulator


def validation(simulator, example, n_features, cfg, rank, device_id, do_autocast = False):

    position, particle_type, material_property, n_particles_per_example, labels = (
        prepare_data(example, device_id)
    )

    # Sample the noise to add to the inputs.
    sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        position, noise_std_last_step=cfg.data.noise_std
    ).to(device_id)
    non_kinematic_mask = (
        (particle_type != cfg.data.kinematic_particle_id).clone().detach().to(device_id)
    )
    sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

    # Do evaluation for the validation data
    device_or_rank = rank if isinstance(device_id, int) else device_id
    # Select the appropriate prediction function
    predict_accelerations = (
        simulator.module.predict_accelerations
        if isinstance(device_id, int)
        else simulator.predict_accelerations
    )
    # Get the predictions and target accelerations
    with torch.no_grad():
        with autocast(dtype = torch.float16, enabled = do_autocast):
            pred_acc, target_acc = predict_accelerations(
                next_positions=labels.to(device_or_rank),
                position_sequence_noise=sampled_noise.to(device_or_rank),
                position_sequence=position.to(device_or_rank),
                nparticles_per_example=n_particles_per_example.to(device_or_rank),
                particle_types=particle_type.to(device_or_rank),
                material_property=(
                    material_property.to(device_or_rank) if n_features == 3 else None
                ),
            )

    # Compute loss
    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

    return loss


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: Config):
    """Train or evaluates the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    use_dist = "LOCAL_RANK" in os.environ
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

    if cfg.mode == "train":
        # If model_path does not exist create new directory.
        if not os.path.exists(cfg.model.path):
            os.makedirs(cfg.model.path, exist_ok=True)

        # Create TensorBoard log directory
        if not os.path.exists(cfg.logging.wandb_dir):
            os.makedirs(cfg.logging.wandb_dir)

        # Train on gpu
        if device == torch.device("cuda"):
            torch.multiprocessing.set_start_method("spawn")
            verbose, world_size = distribute.setup(local_rank)

        # Train on cpu
        else:
            local_rank = None
            world_size = 1
            verbose = True

        train(local_rank, cfg, world_size, device, verbose, use_dist)

    elif cfg.mode in ["valid", "rollout"]:
        # Set device
        world_size = torch.cuda.device_count()
        if cfg.hardware.cuda_device_number is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{int(cfg.hardware.cuda_device_number)}")
        predict(device, cfg)


if __name__ == "__main__":
    main()
