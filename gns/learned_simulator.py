import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict
from gns.Dtype import DType
# from testing import shape_matching
from testing.shape_matching import shape_matching_update

class LearnedSimulator(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""
    
    def __init__(
        self,
        particle_dimensions: int,
        nnode_in: int,
        nedge_in: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        connectivity_radius: float,
        boundaries: np.ndarray,
        norm_stats: dict,
        nparticle_types: int,
        particle_type_embedding_size: int,
        boundary_clamp_limit: float = 1.0,
        device="cpu",
        dtype: DType = DType.SINGLE,
        lazy=False,
        graph_build_freq=1,
        rigid_body_ptype=-1
    ):
        super(LearnedSimulator, self).__init__()

        if dtype == DType.SINGLE:
            self._dtype = torch.float32
            self._use_amp = False
        elif dtype == DType.HALF:
            self._dtype = torch.float16
            self._use_amp = False
        elif dtype == DType.MIXED:
            self._dtype = torch.float32
            self._use_amp = True

        self._lazy = lazy
        self._graph_build_freq = graph_build_freq if self._lazy else None
        self._old_senders = None
        self._old_receivers = None
        self._counter = 0

        self._lower_boundary = boundaries[:, 0][None]
        self._upper_boundary = boundaries[:, 1][None]
        self._connectivity_radius = connectivity_radius
        self._nparticle_types = nparticle_types
        self._boundary_clamp_limit = boundary_clamp_limit
        self._acceleration_mean = norm_stats["acceleration"]["mean"]
        self._acceleration_std = norm_stats["acceleration"]["std"]
        self._velocity_mean = norm_stats["velocity"]["mean"]
        self._velocity_std = norm_stats["velocity"]["std"]
        self._num_edges = 0
        self._device = device
        self._rigid_body_ptype = rigid_body_ptype

        self._particle_type_embedding = nn.Embedding(nparticle_types, particle_type_embedding_size)

        self._encode_process_decode = graph_network.EncodeProcessDecode(
            nnode_in_features=nnode_in,
            nnode_out_features=particle_dimensions,
            nedge_in_features=nedge_in,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            use_amp=self._use_amp
        )

    def forward(self):
        pass

    def reset_graph_state(self):
        self._old_senders = None
        self._old_receivers = None
        self._relative_displacement_with_old_senders = None
        self._counter = 0

    def _compute_graph_connectivity(
        self,
        node_features: torch.tensor,
        nparticles_per_example: torch.tensor,
        radius: float,
        add_self_edges: bool = True,
    ):
        """
        Generate graph edges to all particles within a threshold radius
        """

        # TODO: Is this necessary? If so, it's possible to vectorize this operation
        # batch_ids = torch.cat([
        #     torch.LongTensor([i for _ in range(n)])
        #     for i, n in enumerate(nparticles_per_example)
        # ]).to(self._device)

        edge_index = radius_graph(
            node_features,
            r=radius,
            # batch=batch_ids,
            loop=add_self_edges,
            max_num_neighbors=128,
        )

        receivers, senders = edge_index[0, :], edge_index[1, :]
        self._num_edges = senders.shape[0]
        return receivers, senders

    def _encoder_preprocessor(
        self,
        pos_sequence: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor = None
    ):
        
        nparticles = pos_sequence.shape[0]
        most_recent_pos = pos_sequence[:, -1]
        velocity_sequence = pos_sequence[:, 1:] - pos_sequence[:, :-1]

        if self._lazy is False or (self._counter % self._graph_build_freq == 0):
            senders, receivers = self._compute_graph_connectivity(
                most_recent_pos, 
                nparticles_per_example, 
                self._connectivity_radius
            )
        else:
            senders = self._old_senders
            receivers = self._old_receivers

        node_features = []

        norm_velocity_sequence = (velocity_sequence - self._velocity_mean) / self._velocity_std
        node_features.append(norm_velocity_sequence.view(nparticles, -1).to(dtype=self._dtype))

        distance_to_lower_boundary = (most_recent_pos - self._lower_boundary)
        distance_to_upper_boundary = (self._upper_boundary - most_recent_pos)
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        norm_clipped_distance_to_boundaries = torch.clamp(
            input=distance_to_boundaries / self._connectivity_radius, 
            min=-self._boundary_clamp_limit, 
            max=self._boundary_clamp_limit
        )
        node_features.append(norm_clipped_distance_to_boundaries.to(dtype=self._dtype))

        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        if material_property is not None:
            material_property = material_property.view(nparticles, 1)
            node_features.append(material_property)

        if self._lazy:
            edge_features = _get_edge_features(senders, receivers, most_recent_pos, self._connectivity_radius)
        else:
            norm_relative_displacements = (most_recent_pos[senders] - most_recent_pos[receivers]) / self._connectivity_radius
            norm_relative_distances = torch.norm(norm_relative_displacements, dim=-1, keepdim=True)
            edge_features = [norm_relative_displacements, norm_relative_distances]

        return (
            torch.cat(node_features, -1), 
            torch.stack([senders, receivers]), 
            torch.cat(edge_features, -1)
        )

    def _decoder_postprocessor(
        self,
        norm_acceleration: torch.tensor, 
        pos_sequence: torch.tensor
    ) -> torch.tensor:
        
        acceleration = (norm_acceleration * self._acceleration_std) + self._acceleration_mean

        most_recent_pos = pos_sequence[:, -1]
        most_recent_velocity = most_recent_pos - pos_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration
        new_pos = most_recent_pos + new_velocity

        return new_pos
    
    def _inverse_decoder_postprocessor(
        self,
        next_pos: torch.tensor,
        pos_sequence: torch.tensor
    ) -> torch.tensor:
        previous_pos = pos_sequence[:, -1]
        previous_velocity = previous_pos - pos_sequence[:, -2]
        next_velocity = next_pos - previous_pos
        acceleration = next_velocity - previous_velocity

        norm_acceleration = (acceleration - self._acceleration_mean) / self._acceleration_std
        
        return norm_acceleration

    def predict_positions(
        self,
        current_pos: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor = None,
    ) -> torch.tensor:
        
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_pos,
            nparticles_per_example,
            particle_types,
            material_property
        )
        pred_norm_acceleration = self._encode_process_decode(node_features, edge_index, edge_features)
        if self._rigid_body_ptype > -1:
            pred_norm_acceleration = self.update_rigid_body(current_pos, pred_norm_acceleration)

        next_pos = self._decoder_postprocessor(pred_norm_acceleration, current_pos)

        return next_pos

    def predict_accelerations(
        self,
        next_pos: torch.tensor,
        pos_sequence_noise: torch.tensor,
        pos_sequence: torch.tensor,
        nparticles_per_example: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor = None,
    ):
        """
        Args:
            next_positions: Tensor of shape (nparticles_in_batch, dim) with the positions the model should output given the inputs.
            position_sequence_noise: Tensor of the same shape as `position_sequence` with the noise to apply to each particle.
            position_sequence: A sequence of particle positions. Shape is (nparticles, 6, dim). Includes current + last 5 positions.
            nparticles_per_example: Number of particles per example. Default is 2 examples per batch.
            particle_types: Particle types with shape (nparticles).
            material_property: Friction angle normalized by tan() with shape (nparticles).

        Returns:
            Tensors of shape (nparticles_in_batch, dim) with the predicted and target normalized accelerations.
        """
        
        noisy_pos_sequence = pos_sequence + pos_sequence_noise
        next_pos_adjusted = next_pos + pos_sequence_noise[:, -1]

        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_pos_sequence,
            nparticles_per_example,
            particle_types,
            material_property
        )
        pred_norm_acceleration = self._encode_process_decode(node_features, edge_index, edge_features)
        if self._rigid_body_ptype > -1:
            pred_norm_acceleration = self.update_rigid_body(noisy_pos_sequence, pred_norm_acceleration)

        target_norm_acceleration = self._inverse_decoder_postprocessor(next_pos_adjusted, noisy_pos_sequence)

        return pred_norm_acceleration, target_norm_acceleration
    
    def update_rigid_body(
        self,
        orig_pos: torch.tensor,
        new_pos: torch.tensor,
        particle_types: torch.tensor
    ):
        """
        Args:
            orig_shape: (nparticles, dim)
            new_pos: (nparticles, dim)
            particle_types: Particle types with shape (nparticles)
        """
        mask = (particle_types == self._rigid_body_ptype)
        masked_orig_pos = orig_pos[mask]
        masked_new_pos = new_pos[mask]
        
        return shape_matching_update(masked_orig_pos, masked_new_pos)


@torch.jit.script
def _get_edge_features(
    senders: torch.Tensor,
    receivers: torch.Tensor,
    most_recent_pos: torch.Tensor,
    connectivity_radius: float
) -> torch.Tensor:
    norm_relative_displacements = (most_recent_pos[senders, :] - most_recent_pos[receivers, :]) / connectivity_radius
    norm_relative_distances = torch.norm(norm_relative_displacements, dim=-1, keepdim=True)
    return torch.cat([
        norm_relative_displacements, 
        norm_relative_distances
    ], dim=-1)