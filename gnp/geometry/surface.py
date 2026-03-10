from functools import cached_property

import torch

from ..dataset.patch import PatchData
from .legendre import Legendre2D


class Surface:
    """
    Surface for computing geometric quantities from basis coefficients
    used at evaluation time.

    Parameters
    ----------
    patch_data : Batch
        Patch data in local coordinates from PatchTensor.
    basis_coefficients : torch.Tensor
        Basis coefficients representing the surface (e.g., Legendre coefficients).
    """

    def __init__(
        self,
        patch_data: PatchData,
        basis_coefficients: torch.Tensor,
        use_original: bool = False,
    ):
        self.patch = patch_data
        self.device = patch_data.x.device
        self.x = patch_data.x_original if use_original else patch_data.x
        self.centers = patch_data.centers
        self.xy_scale = patch_data.xy_scale[self.batch]
        self.z_scale = patch_data.z_scale
        self.pca_vectors = patch_data.pca_vectors
        self.coefficients = basis_coefficients
        self.basis = Legendre2D(degree=patch_data.degree)
        self._compute_geometry()

    @property
    def batch(self):
        """
        Batch identifier for which cluster which points belong to.

        Returns
        -------
        torch.Tensor
            Batch indices.
        """

        return self.patch.clusters

    @cached_property
    def local_coordinates(self):
        """
        Returns the local coordinates of the patch data.

        Returns
        -------
        torch.Tensor
            local coordinates of the patch data.
        """
        diff = (self.x - self.centers[self.batch]).unsqueeze(1)
        local_unscaled = (diff @ self.pca_vectors[self.batch].permute(0, 2, 1)).squeeze(
            1
        )
        scaling = torch.cat(
            (self.xy_scale, self.xy_scale, self.z_scale[self.batch]), dim=1
        )
        return local_unscaled / scaling

    def _compute_geometry(self):
        """
        Compute height function and its derivatives for computation of downstream
        geometric quantities.
        """
        self.derivative_scale = torch.reciprocal(
            torch.cat(
                (self.xy_scale.repeat(1, 2), self.xy_scale.pow(2).repeat(1, 3)), dim=1
            )
        )

        scaled_coeffs = self.z_scale * self.coefficients
        self.h = self.basis.evaluate_from_coeffs(
            self.local_coordinates[:, :2], scaled_coeffs[self.batch]
        )
        raw_derivatives = self.basis.derivatives_from_coeffs(
            self.local_coordinates[:, :2], scaled_coeffs[self.batch]
        )
        h_derivatives = self.derivative_scale * raw_derivatives
        (self.h_u, self.h_v, self.h_uv, self.h_uu, self.h_vv) = torch.split(
            h_derivatives, 1, 1
        )

    @property
    def local_coordinate_basis(self) -> torch.Tensor:
        """
        Return the local PCA basis coordinates at each point

        Returns
        -------
        torch.Tensor
            local coordinates of the surface patches
        """
        return self.pca_vectors[self.batch]

    @cached_property
    def xyz_coordinates(self) -> torch.Tensor:
        """
        Compute the xyz coordinates of the surface patches

        Returns
        -------
        torch.Tensor
            xyz coordinates of the surface patches
        """

        pca_x = self.local_coordinates[..., :2].unsqueeze(2)
        pca_basis = self.local_coordinate_basis
        xyz = (
            self.centers[self.batch]
            + self.xy_scale * (pca_x * pca_basis[:, :2]).sum(dim=1)
            + self.h * pca_basis[:, 2]
        )
        return xyz

    @cached_property
    def pca_coordinates(self) -> torch.Tensor:
        """
        Compute the PCA coordinates of the surface patches

        Returns
        -------
        torch.Tensor
            PCA coordinates of the surface patches
        """
        pca_coord = torch.cat(
            (self.local_coordinates[..., :2], self.h / self.z_scale[self.batch]), dim=1
        )
        return pca_coord

    @cached_property
    def tangents_pca(self) -> torch.Tensor:
        """
        Compute the tangents of the surface patches in local coordinates

        Returns
        -------
        torch.Tensor
            tangents of the surface patches
        """
        ones = torch.ones((self.h.shape[0], 1), device=self.device)
        zeros = torch.zeros((self.h.shape[0], 1), device=self.device)
        tangent_u = torch.cat((ones, zeros, self.h_u), dim=-1)
        tangent_v = torch.cat((zeros, ones, self.h_v), dim=-1)
        return torch.stack((tangent_u, tangent_v), dim=1)

    @cached_property
    def tangents(self) -> torch.Tensor:
        """
        Compute the tangents of the surface patches in global coordinates

        Returns
        -------
        torch.Tensor
            tangents of the surface patches
        """
        pca_basis = self.local_coordinate_basis

        return self.tangents_pca @ pca_basis

    @cached_property
    def normals_pca(self) -> torch.Tensor:
        """
        Compute the normals of the surface patches in local coordinates

        Returns
        -------
        torch.Tensor
            normals of the surface patches
        """

        ones = torch.ones((self.h.shape[0], 1), device=self.device)
        normals = torch.cat((-self.h_u, -self.h_v, ones), dim=-1)
        normals = normals / (1 + self.h_u.pow(2) + self.h_v.pow(2)).sqrt()
        return normals

    @cached_property
    def normals(self) -> torch.Tensor:
        """
        Compute the normals of the surface patches in global coordinates

        Returns
        -------
        torch.Tensor
            normals of the surface patches
        """

        pca_basis = self.local_coordinate_basis
        normals = (self.normals_pca.unsqueeze(1) @ pca_basis).squeeze(1)
        normals = normals / normals.norm(dim=1, keepdim=True)
        return normals

    @cached_property
    def metric(self) -> torch.Tensor:
        """
        Compute the metric tensor of the surface patches

        Returns
        -------
        torch.Tensor
            metric tensor of the surface patches
        """

        E = 1 + self.h_u.pow(2)
        F = self.h_u * self.h_v
        G = 1 + self.h_v.pow(2)
        return torch.stack((torch.cat((E, F), dim=1), torch.cat((F, G), dim=1)), dim=1)

    @cached_property
    def shape(self) -> torch.Tensor:
        """
        Compute the shape tensor of the surface patches

        Returns
        -------
        torch.Tensor
            shape tensor of the surface patches
        """

        divisor = (1 + self.h_u.pow(2) + self.h_v.pow(2)).sqrt()
        L = self.h_uu / divisor
        M = self.h_uv / divisor
        N = self.h_vv / divisor
        return torch.stack((torch.cat((L, M), dim=1), torch.cat((M, N), dim=1)), dim=1)

    @cached_property
    def weingarten(self) -> torch.Tensor:
        """
        Compute the Weingarten tensor of the surface patches

        Returns
        -------
        torch.Tensor
            Weingarten tensor of the surface patches
        """

        return torch.linalg.inv(self.metric) @ self.shape

    @cached_property
    def gaussian_curvature(self) -> torch.Tensor:
        """
        Compute the Gaussian curvature tensor of the surface patches

        Returns
        -------
        torch.Tensor
            Gaussian curvature tensor of the surface patches
        """

        return torch.linalg.det(self.weingarten)

    @cached_property
    def mean_curvature(self) -> torch.Tensor:
        """
        Compute the mean curvature tensor of the surface patches

        Returns
        -------
        torch.Tensor
            mean curvature tensor of the surface patches
        """
        return 0.5 * self.weingarten.diagonal(dim1=1, dim2=2).sum(dim=1)

    @cached_property
    def inverse_metric(self) -> torch.Tensor:
        """
        Compute the inverse metric tensor of the surface patches

        Returns
        -------
        torch.Tensor
            inverse metric tensor of the surface patches
        """
        return torch.linalg.inv(self.metric)

    @cached_property
    def inverse_metric_derivatives(self) -> torch.Tensor:
        """
        Compute the derivatives of the inverse metric tensor of the surface patches

        Returns
        -------
        torch.Tensor
            derivatives of the inverse metric tensor of the surface patches
        """
        zero = torch.zeros_like(self.h_u, device=self.device)

        mat1 = torch.cat(
            (
                1 + self.h_v.pow(2),
                -self.h_u * self.h_v,
                -self.h_u * self.h_v,
                1 + self.h_u.pow(2),
            ),
            dim=1,
        ).view(-1, 2, 2)

        mat2_u = torch.cat(
            (
                zero,
                -self.h_uu * self.h_v - self.h_u * self.h_uv,
                -self.h_uu * self.h_v - self.h_u * self.h_uv,
                2 * self.h_u * self.h_uu,
            ),
            dim=1,
        ).view(-1, 2, 2)
        mat2_v = torch.cat(
            (
                2 * self.h_v * self.h_vv,
                -self.h_uv * self.h_v - self.h_u * self.h_vv,
                -self.h_uv * self.h_v - self.h_u * self.h_vv,
                zero,
            ),
            dim=1,
        ).view(-1, 2, 2)
        divisor = 1 + self.h_u.pow(2) + self.h_v.pow(2)

        det_u = (
            -(2 * self.h_u * self.h_uu + 2 * self.h_v * self.h_uv) / divisor.pow(2)
        ).unsqueeze(2)
        det_v = (
            -(2 * self.h_u * self.h_uv + 2 * self.h_v * self.h_vv) / divisor.pow(2)
        ).unsqueeze(2)

        ginv_u = det_u * mat1 + mat2_u / divisor.unsqueeze(2)
        ginv_v = det_v * mat1 + mat2_v / divisor.unsqueeze(2)

        return torch.stack((ginv_u, ginv_v), dim=-1)

    @cached_property
    def det_metric(self) -> torch.Tensor:
        """
        Compute the determinant of the metric tensor of the surface patches

        Returns
        -------
        torch.Tensor
            determinant of the metric tensor of the surface patches
        """
        return 1 + self.h_u.pow(2) + self.h_v.pow(2)

    @cached_property
    def laplace_beltrami_first_terms(self) -> torch.Tensor:
        """
        Compute the first terms of the Laplace-Beltrami operator. These terms
        are multiplied to the second derivatives of the function.

        Returns
        -------
            torch.Tensor
                The first terms of the laplace-beltrami operator.
        """

        return self.det_metric.sqrt() * self.inverse_metric.contiguous().view(-1, 4)

    @cached_property
    def laplace_beltrami_second_terms(self) -> torch.Tensor:
        """
        Compute the second terms of the Laplace-Beltrami operator. These terms
        are multiplied to the first derivatives of the function.

        Returns
        -------
            torch.Tensor
                The second terms of the laplace-beltrami operator.
        """

        det_g = self.det_metric
        uu = (2 * self.h_v * self.h_uv / det_g.sqrt()) - (
            (self.h_u * self.h_uu + self.h_v * self.h_uv)
            * (1 + self.h_v.pow(2))
            / det_g.pow(1.5)
        )
        uv = -((self.h_uu * self.h_v + self.h_u * self.h_uv) / det_g.pow(0.5)) + (
            (self.h_u * self.h_uu + self.h_v * self.h_uv)
            * (self.h_u * self.h_v)
            / det_g.pow(1.5)
        )
        vu = -((self.h_uv * self.h_v + self.h_u * self.h_vv) / det_g.pow(0.5)) + (
            (self.h_u * self.h_uv + self.h_v * self.h_vv)
            * (self.h_u * self.h_v)
            / det_g.pow(1.5)
        )
        vv = (2 * self.h_u * self.h_uv / det_g.sqrt()) - (
            (self.h_u * self.h_uv + self.h_v * self.h_vv)
            * (1 + self.h_u.pow(2))
            / det_g.pow(1.5)
        )
        return torch.cat((uu, uv, vu, vv), dim=1)

    @cached_property
    def laplace_beltrami_basis_terms(self) -> torch.Tensor:
        """Compute the Laplace-Beltrami operator on each of the basis functions.

        Returns
        -------
            torch.Tensor
                (n, 16) tensor containing the Laplace-Beltrami
                operator applied to the basis functions.
        """

        derivatives = self.derivative_scale.unsqueeze(-1) * torch.stack(
            self.basis.evaluate_derivatives(self.local_coordinates[..., :2]),
            dim=1,
        )
        fu, fv, fuv, fuu, fvv = torch.unbind(derivatives, dim=1)

        lb_first = self.laplace_beltrami_first_terms.unsqueeze(-1)
        lb_second = self.laplace_beltrami_second_terms.unsqueeze(-1)

        basis_lb = (
            lb_first * torch.stack((fuu, fuv, fuv, fvv), dim=1)
            + lb_second * torch.stack((fu, fv, fu, fv), dim=1)
        ).sum(dim=1)
        return basis_lb / self.det_metric.sqrt()

    def laplace_beltrami_from_coefficients(
        self, function_coefficients: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Laplace-Beltrami operator from the surface coefficients
        and the function coefficients.

        Parameters
        ----------
        function_coefficients : torch.Tensor
            Basis coefficients representing the function.

        Returns
        -------
        torch.Tensor
            Laplace-Beltrami operator applied to the function.
        """

        lb_values = (
            function_coefficients[self.batch] * self.laplace_beltrami_basis_terms
        ).sum(dim=1, keepdim=True)
        return lb_values
