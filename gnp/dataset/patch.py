from functools import cached_property
from typing import Optional

import torch
import torch_geometric as tg
from torch_geometric.data import Data
from torch_scatter import scatter_add, scatter_max, scatter_mean

from ..utils import QueryTorchGeometric


class PatchData(Data):
    """
     A PyTorch Geometric Data object specialized for patch-based point cloud data.

    This class extends the base `Data` object to include patch-specific
    attributes and a utility for iterating over batches of patches.
    """

    def __init__(self, x: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(x=x, **kwargs)

    @property
    def num_patches(self):
        """Returns the number of patches in the dataset."""
        return self.centers.shape[0] if "centers" in self else 0

    def batch_iterator(self, batch_size: int):
        """
        Create a generator to iterate over batches of patches.

        This method yields smaller `PatchData` objects, each containing a subset
        of the patches. It assumes that the patch data is sorted by `patch_number`.

        Parameters
        ----------
        batch_size : int
            The maximum number of patches in each batch.

        Yields
        ------
        PatchData
            A new `PatchData` object representing a batch of patches.
        """
        num_patches = self.num_patches
        batch_size = min(num_patches, batch_size)
        ptr = torch.zeros(num_patches + 1, dtype=torch.long, device=self.x.device)
        torch.cumsum(self.patch_lens, dim=0, out=ptr[1:])

        for start_idx in range(0, num_patches, batch_size):
            end_idx = min(start_idx + batch_size, num_patches)

            batch_kwargs = {}
            for key, value in self.to_dict().items():
                if isinstance(value, torch.Tensor) and value.shape[0] == num_patches:
                    batch_kwargs[key] = value[start_idx:end_idx]

            edge_start = ptr[start_idx]
            edge_end = ptr[end_idx]
            total_edges = self.patch_indices.shape[0]
            for key, value in self.to_dict().items():
                if isinstance(value, torch.Tensor) and value.shape[0] == total_edges:
                    sliced_val = value[edge_start:edge_end]
                    if key == "patch_number":
                        sliced_val = sliced_val - start_idx

                    batch_kwargs[key] = sliced_val
            yield PatchData(**batch_kwargs)


class PatchTensor:
    """
    Processes a point cloud into a collection of overlapping patches.

    This class handles the entire pipeline of patchifying a point cloud. It
    selects patch centers, finds neighboring points for each patch, computes
    local coordinate systems using PCA, and scales the coordinates. The final
    output is a `PatchData` object ready for use in a model.

    Parameters
    ----------
    data : dict
        A dictionary containing the point cloud data, requires at least an
        'x' key with a tensor of shape (N, 3).
    k : int, optional
        Number of nearest neighbors to consider for various calculations,
        by default 30.
    mode : str, optional
        The mode for center selection ('train', 'test', or 'gmls'),
        by default "test".
    pca : bool, optional
        Whether to use PCA to determine local coordinate systems,
        by default True.
    scale : bool, optional
        Whether to scale the local coordinates, by default True.
    min_z_scale : float, optional
        The minimum value for z-scaling, by default 5e-3.
    basis : str, optional
        The basis to use, by default "legendre".
    basis_degree : int, optional
        The degree of the basis, by default 3.
    num_training_patches : int, optional
        Number of patches to sample in 'train' mode, by default 1024.
    device : str, optional
        The device to perform computations on, by default "cpu".
    """

    def __init__(
        self,
        data: dict,
        k: int = 30,
        mode: str = "test",
        pca: bool = True,
        scale: bool = True,
        min_z_scale: float = 5e-3,
        basis: str = "legendre",
        basis_degree: int = 3,
        num_training_patches: int = 1024,
        device: str = "cpu",
    ):
        self.data = data
        self.x = data["x"].squeeze().to(device)
        self.original_x = data.get("original_x", self.x).squeeze().to(device)

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                self.data[key] = val.to(device)
        self.k = k
        self.mode = mode
        self.pca = pca
        self.scale = scale
        self.min_z_scale = min_z_scale
        self.basis = basis
        self.basis_degree = basis_degree
        self.num_training_patches = num_training_patches
        self.device = device

        self.query = QueryTorchGeometric(x=self.x, device=self.device)
        self.center_indices, self.clusters, self.knn_distances = self.get_centers()
        self.patch_indices, self.patch_number, self.patch_lens = (
            self._patch_data_query()
        )
        self.centers = scatter_mean(
            self.x[self.patch_indices], index=self.patch_number, dim=0
        )

    def get_centers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch method to get patch centers based on the current mode.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - The indices of the center points.
            - The cluster assignment for each point in the cloud.
            - The k-NN distance for each center, used to determine patch radius.
        """
        if self.mode == "train":
            return self.get_train_centers()
        elif self.mode == "test":
            return self.get_test_centers()
        elif self.mode == "gmls":
            return self.get_gmls_centers()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_train_centers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select patch centers by random sampling for training.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor
            Center indices, cluster assignments (dummy), and k-NN distances.
        """
        centers = torch.randperm(self.x.shape[0])[: self.num_training_patches]
        clusters = torch.zeros((self.x.shape[0])).long()
        distances, _ = self.query.query_knn(self.x[centers], k=self.k)

        return (
            centers.to(self.device),
            clusters.to(self.device),
            distances[:, -1].to(self.device),
        )

    def get_test_centers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select patch centers using a greedy covering strategy for testing.

        This method iterates through shuffled points and selects a point as a
        center if it's not already covered by an existing patch, effectively
        creating a set of patches that cover the entire point cloud.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Center indices, cluster assignments, and k-NN distances.
        """
        distances, ind = self.query.query_knn(self.x, k=self.k)
        center_knn_ind = ind[:, : int(0.66 * self.k)].contiguous()

        center_indices = []
        mask = torch.ones(self.x.shape[0], dtype=torch.bool)
        shuffled_indices = torch.randperm(self.x.shape[0])

        for idx in shuffled_indices:
            if mask[idx]:
                center_indices.append(idx)
                neighbors_to_cover = center_knn_ind[idx].flatten()
                mask[neighbors_to_cover] = False
        center_indices = torch.LongTensor(center_indices)
        self.max_patches = center_indices.shape[0] + 1
        knn_ind = ind[center_indices]
        clusters = torch.zeros(self.x.shape[0], dtype=torch.long)
        arange = torch.arange(center_indices.shape[0])
        for j in range(ind.shape[1] - 1, -1, -1):
            clusters[knn_ind[:, j]] = arange
        return (
            center_indices.to(self.device),
            clusters.to(self.device),
            distances[center_indices, -1],
        )

    def get_gmls_centers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select every point as a patch center for GMLS.

        In GMLS mode, a patch is centered at every single point in the cloud.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Center indices, cluster assignments, and k-NN distances.
        """
        distances, _ = self.query.query_knn(self.x, k=self.k)
        return (
            torch.arange(self.x.shape[0], device=self.device),
            torch.arange(self.x.shape[0], device=self.device),
            distances[:, -1],
        )

    def _patch_data_query(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find all points within the radius of each patch center.

        Uses a radius query to find all points belonging to each patch, defined
        by the k-NN distance of the center point.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - The indices of all points belonging to any patch.
            - The patch number (i.e., center index) for each point.
            - The number of points in each patch.
        """
        index_x, index_y = self.query.query_radius(
            x=self.x,
            y=self.x[self.center_indices],
            radius=1.1 * self.knn_distances.max().item(),
            max_num_neighbors=5 * self.k,
        )
        mask = (self.x[index_x] - self.x[self.center_indices[index_y]]).norm(dim=1) < (
            1.1 * self.knn_distances
        )[index_y]
        patch_indices = index_x[mask]
        patch_number = index_y[mask]
        patch_lens = scatter_add(
            torch.ones(patch_number.shape[0], device=self.device, dtype=torch.long),
            patch_number,
        )
        return (
            patch_indices.to(self.device),
            patch_number.to(self.device),
            patch_lens.to(self.device),
        )

    @cached_property
    def tensor_centered(self):
        """
        Dense tensor of patch points, centered by subtracting the patch mean.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_patches, max_points_in_patch, 3).
        """
        patch_tensor, _ = tg.utils.to_dense_batch(
            x=self.x[self.patch_indices], batch=self.patch_number, fill_value=torch.inf
        )
        patch_tensor -= self.centers.unsqueeze(1)
        return torch.nan_to_num(patch_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    @cached_property
    def _pca_data(self):
        """
        Compute PCA vectors and z-scaling for each patch.

        This method calculates the principal component vectors for each patch.
        It aligns the third component with the provided orientation (normals)
        and ensures a right-handed coordinate system. It also computes a
        z-scaling factor based on the standard deviation along the third
        principal component.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor
            A tuple containing:
            - PCA vectors (num_patches, 3, 3).
            - Z-axis scaling factor (num_patches, 1).
        """
        x_centered = self.x[self.patch_indices] - self.centers[self.patch_number]

        outer_prod = x_centered.unsqueeze(2) * x_centered.unsqueeze(1)
        cov_matrices = scatter_add(outer_prod, self.patch_number, dim=0)

        _, S_squared, Vh = torch.linalg.svd(cov_matrices)
        S = S_squared.sqrt()

        pca_vectors = Vh.clone()
        if self.data.get("orientation", None) is not None:
            orientation = self.data.get("orientation")[self.center_indices]
        elif self.data.get("normals", None) is not None:
            orientation = self.data.get("normals")[self.center_indices]
        else:
            orientation = self.x[self.center_indices]

        flip_mask = (orientation * pca_vectors[:, 2]).sum(dim=-1) < 0.0
        pca_vectors[flip_mask, 2] *= -1
        cross_mask = (
            torch.linalg.cross(pca_vectors[:, 0], pca_vectors[:, 1]) * pca_vectors[:, 2]
        ).sum(dim=-1) < 0
        pca_vectors[cross_mask, 1] *= -1
        z_scale = S[:, 2] / (self.patch_lens - 1).sqrt()
        z_mask = z_scale < self.min_z_scale
        z_scale[z_mask] = self.min_z_scale
        z_scale = z_scale.view(-1, 1)
        return pca_vectors, z_scale

    @cached_property
    def _local_coordinate_data(self):
        """
        Compute local coordinates for points in each patch.

        Projects the centered patch points onto the PCA basis and scales them.
        The xy coordinates are scaled by the maximum xy-norm in the patch, and
        the z coordinate is scaled by the pre-computed `z_scale`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - Scaled local coordinates for each point in a patch.
            - The xy-scaling factor for each patch.
        """
        local_unscaled = (
            (
                (
                    self.x[self.patch_indices] - self.centers[self.patch_number]
                ).unsqueeze(1)
            )
            @ self.pca_vectors.permute(0, 2, 1)[self.patch_number]
        ).squeeze(1)
        xy_scale, _ = scatter_max(local_unscaled[:, :2].norm(dim=1), self.patch_number)
        xy_scale = xy_scale.view(-1, 1)
        scaling = torch.cat((xy_scale, xy_scale, self.z_scale), dim=1)
        return local_unscaled / scaling[self.patch_number], xy_scale

    @cached_property
    def _local_coordinate_original_data(self):
        """
        Compute local coordinates for the 'original' points.

        If `original_x` data is provided, this method computes the local
        coordinates for those points using the same transformation (PCA vectors
        and scaling) derived from the primary point cloud.
        """
        x = self.data.get("original_x", None)
        if x is None:
            return None
        else:
            x = x.squeeze(0)

        local_unscaled = (
            ((x[self.patch_indices] - self.centers[self.patch_number]).unsqueeze(1))
            @ self.pca_vectors.permute(0, 2, 1)[self.patch_number]
        ).squeeze(1)

        scaling = torch.cat((self.xy_scale, self.xy_scale, self.z_scale), dim=1)
        return local_unscaled / scaling[self.patch_number]

    @cached_property
    def x_local(self):
        """All points transformed into the local coordinate system of their assigned patch."""
        if self.mode == "train":
            return self.local_coordinates

        local_unscaled = (
            (self.x - self.centers[self.clusters]).unsqueeze(1)
            @ self.pca_vectors[self.clusters].permute(0, 2, 1)
        ).squeeze(1)
        return local_unscaled / self.scaling[self.clusters]

    @cached_property
    def tensor_local(self):
        """Dense tensor of patch points in local coordinates."""
        local_unscaled = (
            self.pca_vectors.unsqueeze(1) @ self.tensor_centered.unsqueeze(-1)
        ).squeeze()
        return local_unscaled / self.scaling.unsqueeze(1)

    @property
    def pca_vectors(self):
        """The PCA vectors (local basis) for each patch."""
        pca_vectors, _ = self._pca_data
        return pca_vectors

    @property
    def z_scale(self):
        """The z-axis scaling factor for each patch."""
        _, z_scale = self._pca_data
        return z_scale

    @property
    def local_coordinates(self):
        """The scaled local coordinates of points within their respective patches."""
        local_coordinates, _ = self._local_coordinate_data
        return local_coordinates

    @property
    def local_coordinates_original(self):
        """The scaled local coordinates of the 'original' points."""
        local_coordinates_original = self._local_coordinate_original_data
        return local_coordinates_original

    @property
    def xy_scale(self):
        """The xy-plane scaling factor for each patch."""
        _, xy_scale = self._local_coordinate_data
        return xy_scale

    @property
    def scaling(self):
        """The combined (x, y, z) scaling vector for each patch."""
        return torch.cat((self.xy_scale, self.xy_scale, self.z_scale), dim=1)

    def as_patch_data(self) -> PatchData:
        """
        Assemble and return the final `PatchData` object.

        This method collects all computed attributes (patch indices, local
        coordinates, PCA vectors, etc.) and any additional data from the input
        dictionary into a single `PatchData` object.

        Returns
        -------
        PatchData
            The fully processed patch data object.
        """
        data_dict = {
            "x": self.x,
            "x_original": self.original_x,
            "mode": self.mode,
            "clusters": self.clusters,
            "centers": self.centers,
            "center_indices": self.center_indices,
            "patch_indices": self.patch_indices,
            "patch_number": self.patch_number,
            "patch_lens": self.patch_lens,
            "local_coordinates": self.local_coordinates,
            "local_coordinates_original": self.local_coordinates_original,
            "pca_vectors": self.pca_vectors,
            "z_scale": self.z_scale,
            "xy_scale": self.xy_scale,
            "degree": self.basis_degree,
        }
        for k, v in self.data.items():
            if k not in data_dict.keys():
                data_dict[k] = v
        return PatchData(**data_dict).to(self.device)
