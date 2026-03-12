import unittest
import torch
from gnp.dataset.patch import PatchTensor


class TestPatchTensor(unittest.TestCase):
    def setUp(self):
        self.pcd = torch.randn(100, 3)
        self.data = {"x": self.pcd}

    def test_patch_tensor_creation(self):
        patch_tensor = PatchTensor(self.data)
        self.assertIsNotNone(patch_tensor)

    def test_as_patch_data(self):
        patch_tensor = PatchTensor(self.data)
        patch_data = patch_tensor.as_patch_data()
        self.assertIsNotNone(patch_data)
        self.assertTrue(hasattr(patch_data, "x"))
        self.assertTrue(hasattr(patch_data, "patch_indices"))
        self.assertTrue(hasattr(patch_data, "patch_number"))

    def test_local_coordinates_computation(self):
        patch_tensor = PatchTensor(self.data)
        patch_data = patch_tensor.as_patch_data()

        self.assertTrue(hasattr(patch_data, "local_coordinates"))
        self.assertIsInstance(patch_data.local_coordinates, torch.Tensor)

    def test_pca_vectors_computation(self):
        patch_tensor = PatchTensor(self.data)
        patch_data = patch_tensor.as_patch_data()

        self.assertTrue(hasattr(patch_data, "pca_vectors"))
        self.assertIsInstance(patch_data.pca_vectors, torch.Tensor)

    def test_geometric_quantities(self):
        patch_tensor = PatchTensor(self.data)
        patch_data = patch_tensor.as_patch_data()

        expected_attrs = ["local_coordinates", "patch_number"]
        for attr in expected_attrs:
            self.assertTrue(hasattr(patch_data, attr), f"Missing attribute: {attr}")


if __name__ == "__main__":
    unittest.main()
