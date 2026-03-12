import unittest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from gnp.models import layers
from gnp.models.gnp import GNP, PatchGNP


class TestLayers(unittest.TestCase):
    def test_get_activation(self):
        self.assertIsInstance(layers.get_activation("ReLU"), nn.ReLU)
        self.assertIsInstance(layers.get_activation("GELU"), nn.GELU)
        self.assertIsInstance(layers.get_activation("LeakyReLU"), nn.LeakyReLU)
        with self.assertRaises(AttributeError):
            layers.get_activation("NonExistentActivation")

    def test_full_kernel(self):
        edge_dim, in_dim, out_dim, neurons = 3, 16, 32, 64
        kernel = layers.FullKernel(edge_dim, in_dim, out_dim, neurons)
        x = torch.randn(10, edge_dim)
        out = kernel(x)
        self.assertEqual(out.shape, (10, in_dim, out_dim))

    def test_block_kernel(self):
        edge_dim, in_dim, out_dim, num_channels, neurons = 3, 16, 32, 4, 64
        kernel = layers.BlockKernel(edge_dim, in_dim, out_dim, num_channels, neurons)
        x = torch.randn(10, edge_dim)
        out = kernel(x)
        self.assertEqual(
            out.shape,
            (10, num_channels, in_dim // num_channels, out_dim // num_channels),
        )

    def test_graph_convolution(self):
        edge_dim, in_dim, out_dim, neurons = 3, 16, 32, 64
        conv = layers.GraphConvolution(edge_dim, in_dim, out_dim, neurons)
        x = torch.randn(5, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, edge_dim)
        out = conv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, out_dim))

    def test_block_factorized_convolution(self):
        edge_dim, in_dim, out_dim, num_channels, neurons = 3, 16, 32, 4, 64
        conv = layers.BlockFactorizedConvolution(
            edge_dim, in_dim, out_dim, num_channels, neurons
        )
        x = torch.randn(5, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, edge_dim)
        out = conv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, out_dim))

    def test_separable_convolution(self):
        in_dim, out_dim, dim_x, neurons = 16, 32, 3, 64
        kernel_args = {}
        conv = layers.SeparableConvolution(
            in_dim, out_dim, dim_x, neurons, "FullKernel", kernel_args
        )
        edge_attr = torch.randn(5, dim_x)
        x = torch.randn(5, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        out = conv(x=x, edge_attr=edge_attr, edge_index=edge_index)
        self.assertEqual(out.shape, (5, out_dim))

    def test_patch_separable_block_factorized_convolution_block(self):
        in_dim, out_dim, dim_x, num_channels, neurons = 16, 32, 3, 4, 64
        conv = layers.PatchSeparableBlockFactorizedConvolutionBlock(
            in_dim, out_dim, dim_x, num_channels, neurons
        )
        x = torch.randn(10, dim_x)
        v = torch.randn(10, in_dim)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
        out = conv(x, v, batch)
        self.assertEqual(out.shape, (10, out_dim))

    def test_convolution_block_full(self):
        in_dim, out_dim, edge_dim = 16, 32, 3
        conv_args = {"neurons": 64}
        block = layers.ConvolutionBlock(
            in_dim, out_dim, edge_dim, "GraphConvolution", conv_args, "ReLU"
        )
        x = torch.randn(5, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, edge_dim)
        out = block(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, out_dim))

    def test_convolution_block_block_factorized(self):
        in_dim, out_dim, edge_dim = 16, 32, 3
        conv_args = {"neurons": 64, "num_channels": 4}
        block = layers.ConvolutionBlock(
            in_dim, out_dim, edge_dim, "BlockFactorizedConvolution", conv_args, "ReLU"
        )
        x = torch.randn(5, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, edge_dim)
        out = block(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, out_dim))

    def test_convolution_block_separable(self):
        in_dim, out_dim, edge_dim = 16, 32, 3
        conv_args = {
            "neurons": 64,
            "kernel_name": "BlockKernel",
            "kernel_args": {"num_channels": 4},
        }
        block = layers.ConvolutionBlock(
            in_dim, out_dim, edge_dim, "SeparableConvolution", conv_args, "ReLU"
        )
        x = torch.randn(5, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(5, edge_dim)
        out = block(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (5, out_dim))


class TestGNPModels(unittest.TestCase):
    def test_gnp_model(self):
        node_dim, edge_dim, out_dim = 3, 3, 1
        layers_list = [16, 32, 16]
        block_args = {"neurons": 64}
        device = "cpu"
        model = GNP(
            node_dim=node_dim,
            edge_dim=edge_dim,
            out_dim=out_dim,
            layers=layers_list,
            conv_name="ConvolutionBlock",
            conv_args={"conv_name": "GraphConvolution", "conv_args": block_args},
            nonlinearity="ReLU",
            skip_connection=True,
            device=device,
        )

        x = torch.randn(10, node_dim)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.randn(2, edge_dim)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        out = model(data)
        self.assertEqual(out.shape, (10, out_dim))

    def test_patch_gnp_model(self):
        node_dim, out_dim = 3, 8
        layers_list = [16, 32, 16]
        num_channels, neurons = 4, 64
        device = "cpu"
        model = PatchGNP(
            node_dim, out_dim, layers_list, num_channels, neurons, "ReLU", device
        )

        x = torch.randn(20, node_dim)
        batch = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

        out = model(x, batch)
        self.assertEqual(out.shape, (2, out_dim))


if __name__ == "__main__":
    unittest.main()
