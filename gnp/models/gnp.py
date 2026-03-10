import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from . import layers as _layers


class GNP(nn.Module):
    """
    General purpose model for Geometric Neural Operators (GNPs).

    This model consists of a lifting layer, multiple graph convolution blocks,
    and a final projection layer.

    Parameters
    ----------
    node_dim : int
        Input dimension of node features.
    edge_dim : int
        Input dimension of edge features.
    out_dim : int
        Output dimension.
    layers : list[int]
        List of hidden dimensions.
    conv_name : str
        Name of the convolution block class to use.
    conv_args : dict
        Arguments for the convolution block.
    nonlinearity : str
        The name of the activation function to use.
    skip_connection: bool
        Whether to use a skip connection for each layer.
    device : str
        The device to place the model on. Defaults to "cuda".
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        layers: list[int],
        conv_name: str,
        conv_args: dict,
        nonlinearity: str,
        skip_connection: bool,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.layers = layers
        self.depth = len(layers) - 1
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.nonlinearity = nonlinearity
        self.lift = nn.Linear(node_dim, layers[0])
        self.proj = nn.Linear(layers[-1], out_dim)

        self.activation = _layers.get_activation(nonlinearity)
        self.num_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        if not hasattr(_layers, conv_name):
            raise ValueError(f"Convolution '{conv_name}' not found in {__name__}")
        self.blocks = nn.ModuleList(
            [
                _layers.ConvolutionBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    edge_dim=edge_dim,
                    nonlinearity=nonlinearity,
                    conv_name=conv_name,
                    conv_args=conv_args,
                    skip=skip_connection,
                )
                for in_dim, out_dim in zip(layers[:-1], layers[1:])
            ]
        )

    def forward(self, data: Data):
        """
        Forward pass.

        Parameters
        ----------
        data : Data
            PyG Data object containing x, edge_index, and edge_attr.

        Returns
        -------
        torch.Tensor
            Output features.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.lift(x)
        for block in self.blocks[:-1]:
            x = block(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.blocks[-1](
            x=x, edge_index=edge_index, edge_attr=edge_attr, use_activation=False
        )
        x = self.proj(x)
        return x


class PatchGNP(nn.Module):
    """
    Geometric Neural Operator for processing point cloud patches.

    Parameters
    ----------
    node_dim : int
        The dimensionality of the input node features (e.g., 3 for xyz).
    out_dim : int
        The dimensionality of the final output vector for each patch.
    layers : list[int]
        A list of integers defining the width of each layer in the network.
        The first element is the width after the initial lifting layer.
    num_channels : int
        The number of channels to use in the 'block' type convolution.
    neurons : int
        The number of neurons in the hidden layers of the MLPs within the
        convolutional layers.
    nonlinearity : str
        The name of the activation function to use.
    device : str
        The device to place the model on. Defaults to "cuda".
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        layers: list[int],
        num_channels: int,
        neurons: int,
        nonlinearity: str,
        device: str = "cuda",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.layer_widths = layers
        self.num_channels = num_channels
        self.neurons = neurons
        self.nonlinearity = nonlinearity
        self.device = device

        self.activation = _layers.get_activation(nonlinearity)
        self.lift = nn.Linear(node_dim, layers[0])
        self.proj = nn.Sequential(
            nn.Linear(layers[-1], 2 * layers[-1]),
            self.activation,
            nn.Linear(2 * layers[-1], out_dim),
        )

        self.convs = nn.ModuleList(
            [
                _layers.PatchSeparableBlockFactorizedConvolutionBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dim_x=node_dim,
                    num_channels=num_channels,
                    neurons=neurons,
                    nonlinearity=nonlinearity,
                )
                for in_dim, out_dim in zip(layers[:-1], layers[1:])
            ]
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        """
        Perform a forward pass on a batch of patches.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates/features (N, node_dim).
        batch : torch.Tensor
            Batch indices (N,).

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_patches, out_dim) containing the output
            vector for each patch.
        """

        v = self.lift(x)
        for i, conv in enumerate(self.convs):
            v = conv(x, v, batch)
            if i < len(self.convs) - 1:
                v = self.activation(v)
        v = global_mean_pool(v, batch)
        return self.proj(v)
