import unittest
import torch
from gnp.estimator import GeometryEstimator

class TestGeometryEstimator(unittest.TestCase):
    def setUp(self):
        self.pcd = torch.randn(100, 3)
        self.orientation = torch.randn(100, 3)
        
    def test_estimator_creation(self):
        estimator = GeometryEstimator(
            pcd=self.pcd,
            orientation=self.orientation,
            model_name="clean_30k"
        )
        self.assertIsNotNone(estimator)
        
    def test_patch_data_generation(self):
        estimator = GeometryEstimator(
            pcd=self.pcd,
            orientation=self.orientation,
            model_name="clean_30k"
        )
        patch_data = estimator.patch_data()
        self.assertIsNotNone(patch_data)
        
    def test_estimate_quantities(self):
        estimator = GeometryEstimator(
            pcd=self.pcd,
            orientation=self.orientation,
            model_name="clean_30k"
        )
        
        quantities = estimator.estimate_quantities(['normals'])
        self.assertIn('normals', quantities)
        self.assertIsInstance(quantities['normals'], torch.Tensor)
        
        quantities = estimator.estimate_quantities(['mean_curvature'])
        self.assertIn('mean_curvature', quantities)
        self.assertIsInstance(quantities['mean_curvature'], torch.Tensor)

    def test_flow_step(self):
        estimator = GeometryEstimator(
            pcd=self.pcd,
            orientation=self.orientation,
            model_name="clean_30k"
        )
        
        result = estimator.flow_step(
            delta_t=0.1,
            subsample_radius=0.5,
            smooth_radius=0.3,
            smooth_x=True
        )
        self.assertIsInstance(result, dict)
        self.assertIn('x', result)
        self.assertIn('normals', result)
        self.assertIn('mean_curvature', result)

    def test_mean_flow(self):
        estimator = GeometryEstimator(
            pcd=self.pcd,
            orientation=self.orientation,
            model_name="clean_30k"
        )
        
        result = estimator.mean_flow(
            num_steps=2,
            save_data_per_step=1,
            delta_t=0.1,
            subsample_radius=0.01,
            smooth_radius=0.1,
            smooth_x=True
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

if __name__ == '__main__':
    unittest.main()
