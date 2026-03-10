from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric as tg
from tqdm import tqdm

from .config import load_config, load_model
from .dataset.patch import PatchData, PatchTensor
from .geometry.surface import Surface
from .utils import smooth_values_by_gaussian, subsample_points_by_radius

MODULE_PATH = Path(__file__).parent


class GeometryEstimator:
    """
    Class used for geometry estimation using the pre-trained PatchGNP.

    Parameters
    ----------
    pcd : torch.Tensor
        Point cloud data (N, 3).
    orientation : torch.Tensor
        Normal vectors for each point (N, 3).
    function_values : torch.Tensor, optional
        Function values defined on the point cloud, by default None.
    model_name : str
        Name of the pre-trained model to use, by default "clean_30k".
        Options: "clean_30k", "clean_50k", "noise_70k", "outlier_50k".
    batch_size : int, optional
        Batch size for processing patches, by default 8192.
    device : str, optional
        Device to run the model on, by default "cpu".
    **data_kwargs :
        Additional keyword arguments for data configuration. These will
        override the default data configurations.
    """

    def __init__(
        self,
        pcd: torch.Tensor,
        orientation: torch.Tensor,
        function_values: Optional[torch.Tensor] = None,
        model_name: str = "clean_30k",
        batch_size: int = 8192,
        device: str = "cpu",
        **data_kwargs: Optional[dict],
    ):
        assert model_name in ["clean_30k", "clean_50k", "noise_70k", "outlier_50k"]
        self.pcd = pcd.to(device)
        self.orientation = orientation.to(device)
        self.device = device

        self.data = {"x": self.pcd, "normals": self.orientation}

        self.config = load_config(
            MODULE_PATH / "model_weights" / model_name / "config.yaml"
        )

        self.model_path = MODULE_PATH / "model_weights" / model_name / "state_dict.pth"
        self.model = load_model(
            config=self.config["model"], model_path=self.model_path, device=device
        )
        self.model.to(device)
        self.batch_size = batch_size

        if function_values is not None:
            self.function_values = function_values.to(device)
            self.data["function_values"] = self.function_values

        for k, v in data_kwargs.items():
            self.config["data"][k] = v

    def patch_data(self, **datakwargs) -> PatchData:
        """
        Create a PatchData object from the point cloud data.

        This method configures and generates patches from the input point cloud
        data, which can then be used by the model for predictions.

        Parameters
        ----------
        **datakwargs :
            Keyword arguments to override data configuration settings.

        Returns
        -------
        PatchData
            A PatchData object containing the point cloud data structured into patches.
        """

        data_config = self.config["data"]
        for k, v in datakwargs.items():
            data_config[k] = v
        return PatchTensor(
            data=self.data, device=self.device, **data_config
        ).as_patch_data()

    def surface_patch(self, patch_data: Optional[PatchData] = None) -> Surface:
        """
        Create a Surface from the input patch data using the predictions from
        the model.

        Parameters
        ----------
        patch_data : PatchData, optional
            Batch containing the input patch data. If patch_data is None defaults
            to self.patch_data()

        Returns
        -------
        Surface
            Surface object.
        """
        if patch_data is None:
            patch_data = self.patch_data()
        surface_coefficients = []
        for pd in patch_data.batch_iterator(self.batch_size):
            x, batch = pd.local_coordinates, pd.patch_number
            with torch.no_grad():
                surface_coefficients.append(self.model(x, batch))
        return Surface(patch_data, torch.cat(surface_coefficients, dim=0))

    def estimate_quantities(self, quantity_names: list[str]) -> dict[str, torch.Tensor]:
        """
        Estimate geometric quantities on the point cloud. This function returns
        a dictionary containing the estimated scalar and/or vector values.

        Parameters
        ----------
        quantity_names : list[str]
            List of quantity names to estimate. Available quantities include:
            'xyz_coordinates', 'normals', 'tangents', 'mean_curvature',
            'gaussian_curvature', 'pca_coordinates', 'normals_pca',
            'tangents_pca', 'metric', 'shape', 'weingarten', 'inverse_metric',
            'inverse_metric_derivatives', 'det_metric',
            'laplace_beltrami_from_coefficients'.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary where keys are the quantity names and values are the
            estimated tensors.
        """

        surface = self.surface_patch()

        output = {}
        for name in quantity_names:
            if hasattr(surface, name):
                output[name] = getattr(surface, name)

        return output

    def flow_step(
        self,
        delta_t: float,
        subsample_radius: float,
        smooth_radius: float,
        smooth_x: bool,
    ) -> dict:
        """
        Perform a single step of mean curvature flow on the point cloud.

        Parameters
        ----------
        delta_t : float
            Time step for the flow.
        subsample_radius : float
            Radius used for subsampling points after the flow step.
        smooth_radius : float
            Radius used for smoothing the mean curvature before the flow step.
        smooth_x : bool
            Whether to smooth the point cloud coordinates before the flow step.

        Returns
        -------
        dict
            A dictionary containing the updated point cloud data ('x'), normals,
            and mean curvature.
        """

        if smooth_x:
            estimate = self.estimate_quantities(["xyz_coordinates"])
            x = estimate["xyz_coordinates"]
            self.pcd = x
            self.data["x"] = x

        estimate = self.estimate_quantities(["normals", "mean_curvature"])
        x = self.pcd
        normals = estimate["normals"]
        mean_curvature = smooth_values_by_gaussian(
            x=x, values=estimate["mean_curvature"], radius=smooth_radius
        )
        new_x = x + delta_t * mean_curvature.view(-1, 1) * normals
        subsampled_indices = subsample_points_by_radius(new_x, subsample_radius)
        new_x = new_x[subsampled_indices]
        new_normals = normals[subsampled_indices]
        mean_curvature = mean_curvature[subsampled_indices]

        new_data = {
            "x": new_x.contiguous(),
            "normals": new_normals.contiguous(),
            "mean_curvature": mean_curvature.contiguous(),
        }

        return new_data

    def mean_flow(
        self,
        num_steps: int,
        save_data_per_step: int,
        delta_t: float,
        subsample_radius: float,
        smooth_radius: float,
        smooth_x: bool,
    ) -> list[dict]:
        """
        Perform mean curvature flow on the point cloud over multiple steps.

        This method iteratively applies the mean curvature flow step and saves
        the state of the point cloud at specified intervals.

        Parameters
        ----------
        num_steps : int
            The total number of flow steps to perform.
        save_data_per_step : int
            The interval at which to save the point cloud data. For example, a
            value of 5 means data is saved every 5 steps.
        delta_t : float
            Time step for each flow step.
        subsample_radius : float
            Radius used for subsampling points after each flow step.
        smooth_radius : float
            Radius used for smoothing the mean curvature before each flow step.
        smooth_x : bool
            Whether to smooth the point cloud coordinates before each flow step.

        Returns
        -------
        list[dict]
            A list of dictionaries. Each dictionary contains the point cloud
            data ('x'), 'normals', and 'mean_curvature' at a saved step.
        """

        save_data = []
        for i in tqdm(range(num_steps)):
            new_data = self.flow_step(
                delta_t=delta_t,
                subsample_radius=subsample_radius,
                smooth_radius=smooth_radius,
                smooth_x=smooth_x,
            )
            self.data = new_data.copy()
            self.pcd = new_data["x"]
            self.orientation = new_data["normals"]
            if not torch.isfinite(new_data["x"]).all():
                print(f"Nan or Infinite detected in Mean Flow! Exiting early at iteration {i}")
                return save_data

            if i % save_data_per_step == 0:
                save_data.append(new_data.copy())

        return save_data

    def gmls_weights(
        self, patch_data: PatchData, mask: torch.Tensor, radius: float = 1.0, p: int = 4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the weights for the generalized moving least squares (GMLS) method.

        Parameters
        ----------
        patch_data : PatchData
            PatchData object containing coordinate data.
        mask : torch.Tensor
            A boolean mask indicating which points in the patch data to use.
        radius : float, optional
            Radius to truncate the weight function, by default 1.0.
        p : int, optional
            The exponent for the weight function, by default 4.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - The weight matrices for each patch (batch_size, 1, num_points).
            - The dense mask used to create the dense batch.
        """
        uv = patch_data.local_coordinates[mask, :2]
        dists = uv.norm(dim=1)
        dists_dense, mask = tg.utils.to_dense_batch(
            x=dists, batch=patch_data.patch_number[mask], fill_value=torch.inf
        )
        weights = F.relu(1 - dists_dense / radius).pow(p)

        return weights.unsqueeze(1), mask

    def laplace_beltrami_legendre_blocks(self, surface: Surface) -> torch.Tensor:
        """
        Compute the Laplace-Beltrami operator of Legendre basis functions.

        Parameters
        ----------
        surface : Surface
            Surface object containing patch information and basis functions.

        Returns
        -------
        torch.Tensor
            A tensor containing the Laplace-Beltrami operator applied
        """

        return surface.laplace_beltrami_basis_terms

    def legendre_blocks(self, surface: Surface, mask: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Legendre basis functions for each point in the patches.

        This method evaluates the Legendre basis functions at the local `uv`
        coordinates of the points specified by the mask and returns them as a
        dense tensor, batched by patch.

        Parameters
        ----------
        surface : Surface
            Surface object containing patch data and basis functions.
        mask : torch.Tensor
            A boolean mask to select which points' local coordinates to use for
            the evaluation.

        Returns
        -------
        torch.Tensor
            A dense tensor of Legendre basis function evaluations, with shape
            (num_patches, max_points_in_patch, num_basis_functions).
        """
        uv = surface.patch.local_coordinates[mask, :2]
        legendre_values = surface.basis.evaluate(uv)
        legendre_blocks, _ = tg.utils.to_dense_batch(
            x=legendre_values, batch=surface.patch.patch_number[mask], fill_value=0
        )

        return legendre_blocks

    def stiffness_matrix_gmls(
        self,
        drop_ratio: float = 0.1,
        radius: float = 1.0,
        p: int = 4,
        remove_outliers: bool = False,
        outlier_threshold: float = 0.2,
    ) -> tuple[sp.csr_array, torch.Tensor, torch.Tensor]:
        """
        Compute the stiffness matrix using the generalized moving least squares (GMLS) method.

        Parameters
        ----------
            drop_ratio: float, optional
                Ratio of points to drop. Defaults to 0.1.
            radius: float, optional
                Radius to truncate weight function. Defaults to 1..
            p: int, optional
                Degree p of the weight function. Defaults to 4.
            remove_outliers: bool, optional
                Whether to remove outliers. Defaults to False.
            outlier_threshold: float, optional
                The threshold used to determine which points are labeled as outliers.
                Only used if remove_outliers is True.

        Returns
        -------
            sp.csr_array
                Stiffness matrix for solving the Laplace-Beltrami PDE
            torch.Tensor
                Mask for which values to solve collocation problem on.
            torch.Tensor
                Mask for points that are removed in smoothing. If remove_outliers is False
                then the mask will be all true.
        """

        if remove_outliers:
            outputs = self.estimate_quantities(["local_coordinates", "pca_coordinates"])
            outlier_mask = (
                outputs["local_coordinates"] - outputs["pca_coordinates"]
            ).norm(dim=1) < outlier_threshold
            for k, v in self.data.items():
                self.data[k] = v[outlier_mask]
            self.pcd = self.pcd[outlier_mask]
            self.orientation = self.orientation[outlier_mask]
        else:
            outlier_mask = torch.ones(
                self.pcd.shape[0], dtype=torch.bool, device=self.pcd.device
            )
        if drop_ratio > 0.0:
            drop_inds = tg.nn.fps(self.pcd, ratio=drop_ratio).to(self.device)
        else:
            drop_inds = torch.LongTensor([])
        collocation_mask = torch.ones(
            self.pcd.shape[0], dtype=torch.bool, device=self.device
        )
        collocation_mask[drop_inds] = False

        patch_data = self.patch_data(mode="gmls")
        coord_mask = collocation_mask[patch_data.patch_indices]
        surface = self.surface_patch(patch_data)
        weights, tensor_mask = self.gmls_weights(patch_data, coord_mask, radius, p)
        legendre = self.legendre_blocks(surface, coord_mask)
        lb = self.laplace_beltrami_legendre_blocks(surface)

        ls_solutions = torch.linalg.lstsq(
            (legendre.permute(0, 2, 1) * weights) @ legendre,
            legendre.permute(0, 2, 1) * weights,
        ).solution

        stiffness_values = (-lb.unsqueeze(1) @ ls_solutions).squeeze(1)[tensor_mask]

        _, patch_indices_reindexed = torch.unique(
            patch_data.patch_indices[coord_mask], return_inverse=True
        )
        stiffness_indices = torch.stack(
            [
                patch_data.patch_number.flatten()[coord_mask],
                patch_indices_reindexed,
            ],
            dim=0,
        )
        stiffness = sp.coo_matrix(
            (
                stiffness_values.cpu().numpy(),
                stiffness_indices.cpu().numpy().astype(np.int32),
            )
        ).tocsr()
        return stiffness, collocation_mask, outlier_mask
