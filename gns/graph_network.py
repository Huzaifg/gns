from typing import List, Optional
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: Optional[int] = None,
    output_activation: nn.Module = nn.Identity,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    layers = []
    layer_sizes = [input_size] + hidden_layer_sizes + ([output_size] if output_size else [])
    
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(activation() if i < len(layer_sizes) - 2 else output_activation())
    
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    """Graph network encoder."""

    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        nedge_out_features: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super().__init__()
        self.node_fn = nn.Sequential(
            build_mlp(
                nnode_in_features,
                [mlp_hidden_dim] * nmlp_layers,
                nnode_out_features,
            ),
            nn.LayerNorm(nnode_out_features),
        )
        self.edge_fn = nn.Sequential(
            build_mlp(
                nedge_in_features,
                [mlp_hidden_dim] * nmlp_layers,
                nedge_out_features,
            ),
            nn.LayerNorm(nedge_out_features),
        )

    def forward(self, x: torch.Tensor, edge_features: torch.Tensor):
        return self.node_fn(x), self.edge_fn(edge_features)

class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super().__init__(aggr="add")
        self.node_fn = nn.Sequential(
            build_mlp(
                nnode_in + nedge_out,
                [mlp_hidden_dim] * nmlp_layers,
                nnode_out,
            ),
            nn.LayerNorm(nnode_out),
        )
        self.edge_fn = nn.Sequential(
            build_mlp(
                nnode_in * 2 + nedge_in,
                [mlp_hidden_dim] * nmlp_layers,
                nedge_out,
            ),
            nn.LayerNorm(nedge_out),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ):
        x_residual, edge_features_residual = x, edge_features
        x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        return x + x_residual, edge_features + edge_features_residual

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        return self.edge_fn(torch.cat([x_i, x_j, edge_features], dim=-1))

    def update(self, x_updated: torch.Tensor, x: torch.Tensor, edge_features: torch.Tensor):
        return self.node_fn(torch.cat([x_updated, x], dim=-1)), edge_features

class Processor(nn.Module):
    """The Processor computes interactions among nodes via learned message-passing."""

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super().__init__(aggr="max")
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                nnode_in=nnode_in,
                nnode_out=nnode_out,
                nedge_in=nedge_in,
                nedge_out=nedge_out,
                nmlp_layers=nmlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            )
            for _ in range(nmessage_passing_steps)
        ])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ):
        for gnn in self.gnn_stacks:
            x, edge_features = gnn(x, edge_index, edge_features)
        return x, edge_features

class Decoder(nn.Module):
    """The Decoder extracts the dynamics information from the nodes of the final latent graph."""

    def __init__(
        self, nnode_in: int, nnode_out: int, nmlp_layers: int, mlp_hidden_dim: int
    ):
        super().__init__()
        self.node_fn = build_mlp(
            nnode_in, [mlp_hidden_dim] * nmlp_layers, nnode_out
        )

    def forward(self, x: torch.Tensor):
        return self.node_fn(x)

class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        use_amp: bool,
        device="cuda"
    ):
        super().__init__()
        self._encoder = Encoder(
            nnode_in_features=nnode_in_features,
            nnode_out_features=latent_dim,
            nedge_in_features=nedge_in_features,
            nedge_out_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._use_amp = use_amp
        self.device = device

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ):
        """
        Args:
            x: Particle state representation as a torch tensor with shape (nparticles, nnode_in_features)
            edge_index: A torch tensor list of source and target nodes with shape (2, nedges)
            edge_features: Edge features as a torch tensor with shape (nedges, nedge_in_features)
          
        Returns:
            x: Particle state representation as a torch tensor with shape (nparticles, nnode_out_features)
        """
        with torch.autocast(self.device, dtype=torch.float16, enabled=self._use_amp):
            x, edge_features = self._encoder(x, edge_features)
            x, edge_features = self._processor(x, edge_index, edge_features)
            x = self._decoder(x)
        return x