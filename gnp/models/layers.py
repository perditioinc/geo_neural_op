import sys

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean


def get_activation(name: str) -> nn.Module:
    """Helper to retrieve activation functions."""
    try:
        activation = getattr(nn, name)()
    except AttributeError:
        raise AttributeError(
            f"Invalid Activation: module torch.nn has no attribute {name}"
        )
    else:
        return activation


class FullKernel(nn.Module):
    """
    An MLP for a graph convolution.

    This module maps edge features to a dense weight matrix for graph convolution.

    Parameters
    ----------
    edge_dim : int
        Dimension of the input edge features (e.g., 3 for xyz).
    in_dim : int
        Dimension of the input node features for the convolution.
    out_dim : int
        Dimension of the output node features for the convolution.
    neurons : int
        Number of neurons in the hidden layers of the MLP.
    nonlinearity : str
        Nonlinearity to use in the hidden layers, by default 'ReLU'.
    """

    def __init__(
        self,
        edge_dim: int,
        in_dim: int,
        out_dim: int,
        neurons: int,
        nonlinearity: str = "ReLU",
    ):
        super().__init__()
        self.d_x = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neurons = neurons

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.d_x, neurons),
                nn.Linear(neurons, neurons),
                nn.Linear(neurons, self.in_dim * self.out_dim),
            ]
        )
        self.activation = get_activation(nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel weights from the input features.

        Parameters
        ----------
        x : torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The computed kernel of shape (..., in_dim, out_dim).
        """
        z = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            z = self.activation(layer(z))
        z = self.layers[-1](z)
        return z.reshape(-1, self.in_dim, self.out_dim)


class BlockKernel(nn.Module):
    """
    An MLP that computes a block-factorized kernel for a graph convolution.

    This module takes edge features as input and produces a weight matrix
    (kernel) that is factorized into smaller blocks.
    This is used to reduce the number of parameters in the convolution.
    """

    def __init__(
        self,
        edge_dim: int,
        in_dim: int,
        out_dim: int,
        num_channels: int,
        neurons: int,
        nonlinearity: str = "ReLU",
    ):
        """
        Initialize the BlockKernel MLP.

        Parameters
        ----------
        edge_dim : int
            Dimension of the input edge features (e.g., 3 for xyz).
        in_dim : int
            Dimension of the input node features for the convolution.
        out_dim : int
            Dimension of the output node features for the convolution.
        num_channels : int
            Number of blocks (channels) to factorize the kernel into.
        neurons : int
            Number of neurons in the hidden layers of the MLP.
        nonlinearity : str
            Nonlinearity to use in the hidden layers, by default 'ReLU'.
        """
        super().__init__()
        self.d_x = edge_dim
        self.channels = num_channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_in = in_dim // num_channels
        self.head_out = out_dim // num_channels
        self.neurons = neurons
        self.out_dim = self.channels * self.head_in * self.head_out

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.d_x, neurons),
                nn.Linear(neurons, neurons),
                nn.Linear(neurons, self.out_dim),
            ]
        )
        self.activation = get_activation(nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel weights from the input features.

        Parameters
        ----------
        x : torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The computed kernel, reshaped into a block-diagonal-like structure
            of shape (..., num_channels, head_in, head_out).
        """
        z = x
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        z = self.layers[-1](z)
        return z.view(-1, self.channels, self.head_in, self.head_out)


class GraphConvolution(MessagePassing):
    """
    Standard graph convolution layer with a continuous kernel using mean aggregation.

    Parameters
    ----------
    edge_dim : int
        Dimension of edge features.
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    neurons : int
        Hidden neurons in the kernel MLP.
    nonlinearity : str
        Activation function, by default 'ReLU'.
    """

    def __init__(
        self,
        edge_dim: int,
        in_dim: int,
        out_dim: int,
        neurons: int,
        nonlinearity: str = "ReLU",
    ):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn = FullKernel(edge_dim, in_dim, out_dim, neurons, nonlinearity)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph connectivity (2, num_edges).
        edge_attr : torch.Tensor
            Edge features.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Construct messages using the kernel."""
        weights = self.nn(edge_attr)
        return torch.matmul(x_j.unsqueeze(-2), weights).squeeze(-2)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node embeddings."""
        return aggr_out


class BlockFactorizedConvolution(MessagePassing):
    """
    A graph convolution layer using a block-factorized kernel.

    Parameters
    ----------
    edge_dim : int
        Dimension of edge features.
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    num_channels : int
        Number of blocks/channels for factorization.
    neurons : int
        Hidden neurons in the kernel MLP.
    nonlinearity : str
        Activation function, by default 'ReLU'.
    """

    def __init__(
        self,
        edge_dim: int,
        in_dim: int,
        out_dim: int,
        num_channels: int,
        neurons: int,
        nonlinearity: str = "ReLU",
    ):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_in = in_dim // num_channels
        self.channels = num_channels
        self.nn = BlockKernel(
            edge_dim, in_dim, out_dim, num_channels, neurons, nonlinearity
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph connectivity.
        edge_attr : torch.Tensor
            Edge features.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Construct messages using the block kernel."""
        weights = self.nn(edge_attr)
        return torch.matmul(x_j.view(-1, self.channels, 1, self.head_in), weights).view(
            -1, self.out_dim
        )

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node embeddings."""
        return aggr_out


class SeparableConvolution(nn.Module):
    """
    A separable convolution layer that evaluates a separable kernel on nodes
    instead of edges.

    This layer applies convolution in two steps: convolution using a kernel
    that only depends on the source nodes followed by multiplication by a kernel
    that only depends on the target node.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    edge_dim : int
        Dimension of spatial/edge features.
    neurons : int
        Number of neurons in the kernel MLP.
    kernel_name : str
        Name of the kernel class to use (e.g., 'BlockKernel').
    kernel_args : dict
        Additional arguments for the kernel class.
    nonlinearity : str
        Activation function, by default 'ReLU'.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        neurons: int,
        kernel_name: str,
        kernel_args: dict,
        nonlinearity: str = "ReLU",
    ):
        super().__init__()
        self.dim_in = in_dim
        self.dim_out = out_dim
        module = sys.modules[__name__]
        if not hasattr(module, kernel_name):
            raise ValueError(f"Kernel '{kernel_name}' not found in {__name__}")
        if kernel_name == "BlockKernel":
            self.matmul = self.block_matmul
            self.channels: int = kernel_args.get("num_channels", 1)
        else:
            self.matmul = self.full_matmul

        kernel_cls = getattr(module, kernel_name)
        self.kernel1 = kernel_cls(
            edge_dim=edge_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            neurons=neurons,
            nonlinearity=nonlinearity,
            **kernel_args,
        )
        self.kernel2 = kernel_cls(
            edge_dim=edge_dim,
            in_dim=out_dim,
            out_dim=out_dim,
            neurons=neurons,
            nonlinearity=nonlinearity,
            **kernel_args,
        )
        self.mix = (
            nn.Linear(out_dim, out_dim)
            if kernel_name == "BlockKernel"
            else nn.Identity()
        )

    def full_matmul(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with weigths from the FullKernel

        Parameters
        ----------
        x : torch.Tensor
            Features to be multiplied against kernel
        weights : torch.Tensor
            Kernel weights

        Returns
        -------
        torch.Tensor
            Matrix multiplication product of x and weights.
        """
        return torch.matmul(x.unsqueeze(-2), weights).squeeze(-2)

    def block_matmul(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with weigths from the BlockKernel

        Parameters
        ----------
        x : torch.Tensor
            Features to be multiplied against kernel
        weights : torch.Tensor
            Kernel weights

        Returns
        -------
        torch.Tensor
            Matrix multiplication product of x and weights.
        """
        _, N = x.shape
        return torch.matmul(
            x.view(-1, self.channels, 1, int(N / self.channels)), weights
        ).flatten(1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (N, in_dim).
        edge_index : torch.Tensor
            Graph connectivity.
        edge_attr : torch.Tensor
            Spatial node features of shape (N, edge_dim).

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        weights_y = self.kernel1(edge_attr.detach())
        weights_x = self.kernel2(edge_attr)
        ky_v = self.matmul(x, weights_y)
        v_conv_y = scatter_mean(
            ky_v[edge_index[1]], index=edge_index[0], dim=0, dim_size=x.shape[0]
        )
        v_conv = self.mix(self.matmul(v_conv_y, weights_x))

        return v_conv


class PatchSeparableBlockFactorizedConvolutionBlock(nn.Module):
    """
    A separable graph convolution using a block-factorized kernel specifically
    meant for the PatchGNP.

    This convolutional layer performs a two-step message passing operation.
    It first aggregates messages from neighbors and then applies
    a second transformation based on the point's own features.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    dim_x : int
        Dimension of spatial features.
    num_channels : int
        Number of channels for block factorization.
    neurons : int
        Hidden neurons in MLPs.
    nonlinearity : str
        Activation function, by default 'ReLU'.
    skip : bool
        Whether to use a residual skip connection, by default True.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dim_x: int,
        num_channels: int,
        neurons: int,
        nonlinearity: str = "ReLU",
        skip: bool = True,
    ):
        super().__init__()
        self.head_in = int(in_dim // num_channels)
        self.head_out = int(out_dim // num_channels)
        self.dim_out = out_dim
        self.skip = skip
        self.activation = get_activation(nonlinearity)
        self.channels = num_channels
        self.kernel1 = BlockKernel(
            edge_dim=dim_x,
            in_dim=in_dim,
            out_dim=out_dim,
            num_channels=num_channels,
            neurons=neurons,
            nonlinearity=nonlinearity,
        )
        self.kernel2 = BlockKernel(
            edge_dim=dim_x,
            in_dim=out_dim,
            out_dim=out_dim,
            num_channels=num_channels,
            neurons=neurons,
            nonlinearity=nonlinearity,
        )
        self.linear = nn.Linear(out_dim, out_dim)
        if in_dim != out_dim:
            self.skip_layer = nn.Linear(in_dim, out_dim)
        else:
            self.skip_layer = nn.Identity()

    def forward(
        self, x: torch.Tensor, v: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the forward pass for the separable convolution.

        Parameters
        ----------
        x : torch.Tensor
            The spatial/edge features (e.g., local coordinates).
        v : torch.Tensor
            The input node features.
        batch : torch.Tensor
            A tensor mapping each node to its corresponding patch index.

        Returns
        -------
        torch.Tensor
            The updated node features after convolution.
        """
        weights_y = self.kernel1(x.detach())
        weights_x = self.kernel2(x)
        ky_v = torch.matmul(v.view(-1, self.channels, 1, self.head_in), weights_y).view(
            -1, self.dim_out
        )
        v_conv_y = scatter_mean(ky_v, index=batch, dim=0)
        v_conv = torch.matmul(
            v_conv_y[batch].view(-1, self.channels, 1, self.head_out), weights_x
        ).view(-1, self.dim_out)

        if self.skip:
            v = self.skip_layer(v) + self.linear(v_conv)
        else:
            v = v_conv

        return v


class ConvolutionBlock(nn.Module):
    """
    A unified block combining a graph convolution, residual connection, and activation.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    edge_dim : int
        Edge feature dimension.
    conv_name : str
        Name of the convolution layer class.
    conv_args : dict
        Arguments for the convolution layer.
    nonlinearity : str
        Activation function name.
    skip : bool
        Whether to use a skip connection, by default True.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        conv_name: str,
        conv_args: dict,
        nonlinearity: str,
        skip: bool = True,
    ):
        super().__init__()
        self.activation = get_activation(nonlinearity)
        self.skip = skip
        module = sys.modules[__name__]
        if not hasattr(module, conv_name):
            raise ValueError(f"Convolution '{conv_name}' not found in {__name__}")

        conv_cls = getattr(module, conv_name)
        self.conv = conv_cls(
            in_dim=in_dim,
            out_dim=out_dim,
            edge_dim=edge_dim,
            nonlinearity=nonlinearity,
            **conv_args,
        )
        self.mix = (
            nn.Linear(out_dim, out_dim)
            if conv_name == "BlockFactorizedConvolution"
            else nn.Identity()
        )

        if in_dim != out_dim:
            self.skip_layer = nn.Linear(in_dim, out_dim)
        else:
            self.skip_layer = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        use_activation: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph connectivity.
        edge_attr : torch.Tensor
            Edge features.
        use_activation : bool
            Whether to apply activation at the end, by default True.

        Returns
        -------
        torch.Tensor
            Updated features.
        """
        z = self.mix(self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr))

        if self.skip:
            z = z + self.skip_layer(x)

        if use_activation:
            z = self.activation(z)

        return z
