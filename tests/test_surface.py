import unittest

import torch

from gnp.dataset.patch import PatchTensor
from gnp.geometry.surface import Surface


class TestSurface(unittest.TestCase):
    def setUp(self):
        self.pcd = torch.randn(100, 3)
        self.orientation = torch.randn(100, 3)
        self.data = {"x": self.pcd, "normals": self.orientation}

        patch_tensor = PatchTensor(self.data)
        self.patch_data = patch_tensor.as_patch_data()

        self.basis_coefficients = torch.randn(self.patch_data.num_patches, 16)

    def test_surface_creation(self):
        surface = Surface(self.patch_data, self.basis_coefficients)
        self.assertIsNotNone(surface)

    def test_xyz_coordinates_computation(self):
        surface = Surface(self.patch_data, self.basis_coefficients)
        xyz_coords = surface.xyz_coordinates
        self.assertIsInstance(xyz_coords, torch.Tensor)
        self.assertEqual(xyz_coords.shape, self.pcd.shape)

    def test_normals_computation(self):
        surface = Surface(self.patch_data, self.basis_coefficients)
        normals = surface.normals
        self.assertIsInstance(normals, torch.Tensor)
        self.assertEqual(normals.shape, self.orientation.shape)
        self.assertTrue(
            torch.isclose(normals.norm(dim=1), torch.ones(normals.size(0))).all()
        )

    def test_curvature_computation(self):
        surface = Surface(self.patch_data, self.basis_coefficients)

        mean_curvature = surface.mean_curvature
        self.assertIsInstance(mean_curvature, torch.Tensor)

        gaussian_curvature = surface.gaussian_curvature
        self.assertIsInstance(gaussian_curvature, torch.Tensor)

    def test_metric_computation(self):
        surface = Surface(self.patch_data, self.basis_coefficients)
        metric = surface.metric
        self.assertIsInstance(metric, torch.Tensor)
        self.assertEqual(metric.shape[0], self.pcd.shape[0])
        self.assertEqual(metric.shape[1:], (2, 2))

    def test_shape_computation(self):
        surface = Surface(self.patch_data, self.basis_coefficients)
        shape = surface.shape
        self.assertIsInstance(shape, torch.Tensor)
        self.assertEqual(shape.shape[0], self.pcd.shape[0])
        self.assertEqual(shape.shape[1:], (2, 2))


if __name__ == "__main__":
    unittest.main()
