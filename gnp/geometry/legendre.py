from typing import Optional

import torch
from scipy.special import legendre


class Legendre1D:
    """
    Computes 1D Legendre polynomials and their derivatives.

    This class pre-computes the coefficients of Legendre polynomials up to a
    specified degree and provides methods to evaluate the polynomials and their
    derivatives on batches of input data.

    Parameters
    ----------
    degree : int
        The maximum degree of the Legendre polynomials, by default 3.
    """

    def __init__(self, degree: int = 3):
        self.degree = degree
        self.legendres = [torch.Tensor(legendre(i).coef) for i in range(degree + 1)]

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Legendre polynomials P_0(x), ..., P_degree(x).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            A tensor of shape (*x.shape, degree + 1) containing the evaluated
            polynomials for each input value.
        """

        xs = torch.stack([x.pow(i) for i in range(self.degree, -1, -1)], dim=-1)
        xs = torch.stack(
            [
                (self.legendres[i].to(x.device) * xs[..., -i - 1 :]).sum(dim=-1)
                for i in range(self.degree + 1)
            ],
            dim=-1,
        )
        return xs

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the first derivative of the Legendre polynomials.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            A tensor containing the first derivatives.
        """

        return self.derivative(x, order=1)

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the second derivative of the Legendre polynomials.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            A tensor containing the second derivatives.
        """

        return self.derivative(x, order=2)

    def derivative(self, x: torch.Tensor, order: int) -> torch.Tensor:
        """
        Compute the n-th order derivative of the Legendre polynomials.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.
        order : int
            The order of the derivative to compute.

        Returns
        -------
        torch.Tensor
            A tensor of shape (*x.shape, degree + 1) containing the evaluated
            derivatives.
        """

        assert order > 0

        xs = torch.stack(
            [
                torch.prod(torch.arange(i + 1 - order, i + 1)) * x.pow(i - order)
                if i - order >= 0
                else torch.zeros_like(x)
                for i in range(self.degree, -1, -1)
            ],
            dim=-1,
        )
        derivatives = torch.stack(
            [
                (self.legendres[i].to(x.device) * xs[..., -i - 1 :]).sum(dim=-1)
                for i in range(self.degree + 1)
            ],
            dim=-1,
        )
        return derivatives


class Legendre2D:
    """
    Computes a 2D Legendre basis from the tensor product of 1D polynomials.

    This class constructs a basis of 2D functions L_{i,j}(u,v) = P_i(u) * P_j(v)
    and provides methods to evaluate the basis functions and their derivatives.

    Parameters
    ----------
    degree : int
        The maximum degree for the 1D Legendre polynomials used to
        construct the 2D basis.
    """

    def __init__(self, degree: int):

        super().__init__()
        self._degree = degree
        self.legendre_1d = Legendre1D(degree)

    @property
    def degree_indices(self):
        """
        Get the pairs of degrees (i, j) for each 2D basis function.

        Returns
        -------
        torch.Tensor
            A tensor of shape (2, num_components) where each column is a
            pair of degrees [i, j].
        """

        indices = torch.nonzero(torch.ones(self._degree + 1, self._degree + 1)).T
        return indices

    @property
    def num_components(self):
        """The total number of basis functions in the 2D basis."""

        return self.degree_indices.shape[1]

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all 2D basis functions L_{i,j}(u,v) at given (u,v) coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2) containing (u,v) coordinates.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., num_components) containing the evaluated
            2D basis functions.
        """

        u = self.legendre_1d.evaluate(x[..., 0])[..., self.degree_indices[0]]
        v = self.legendre_1d.evaluate(x[..., 1])[..., self.degree_indices[1]]
        return u * v

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of each 2D basis function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2) containing (u,v) coordinates.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., 2, num_components) where the second-to-last
            dimension corresponds to the partial derivatives [dL/du, dL/dv].
        """

        u = self.legendre_1d.evaluate(x[..., 0])[..., self.degree_indices[0]]
        v = self.legendre_1d.evaluate(x[..., 1])[..., self.degree_indices[1]]

        du = self.legendre_1d.derivative(x[..., 0], order=1)[
            ..., self.degree_indices[0]
        ]
        dv = self.legendre_1d.derivative(x[..., 1], order=1)[
            ..., self.degree_indices[1]
        ]
        return torch.stack((du * v, u * dv), dim=-2)

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian matrix for each 2D basis function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2) containing (u,v) coordinates.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., 2, 2, num_components) representing the
            Hessian matrix [[d2L/du2, d2L/dudv], [d2L/dvdu, d2L/dv2]] for each
            basis function.
        """

        u = self.legendre_1d.evaluate(x[..., 0])[..., self.degree_indices[0]]
        v = self.legendre_1d.evaluate(x[..., 1])[..., self.degree_indices[1]]

        du = self.legendre_1d.derivative(x[..., 0], order=1)[
            ..., self.degree_indices[0]
        ]
        dv = self.legendre_1d.derivative(x[..., 1], order=1)[
            ..., self.degree_indices[1]
        ]

        ddu = self.legendre_1d.derivative(x[..., 0], order=2)[
            ..., self.degree_indices[0]
        ]
        ddv = self.legendre_1d.derivative(x[..., 1], order=2)[
            ..., self.degree_indices[1]
        ]

        hessian = torch.stack((ddu * v, du * dv, du * dv, u * ddv), dim=-2)
        return hessian.reshape(*x.shape[:-1], 2, 2, self.num_components)

    def evaluate_from_coeffs(
        self,
        xy_data: torch.Tensor,
        coeffs: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate a function defined by coefficients on the Legendre basis.

        The function is f(u,v) = sum(coeffs * L_{i,j}(u,v)).

        Parameters
        ----------
        xy_data : torch.Tensor
            Input tensor of shape (N, 2) with (u,v) coordinates.
        coeffs : torch.Tensor
            Coefficients for the Legendre basis. Can be shape (M, num_components)
            if using batch, or (N, num_components) otherwise.
        batch : torch.Tensor, optional
            A tensor of shape (N,) mapping each point in xy_data to a set of
            coefficients. If None, coeffs must be shape (N, num_components).
            Defaults to None.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 1) with the evaluated function values.
        """

        legendre_values = self.evaluate(xy_data)
        return self.compute_from_coeffs(legendre_values, coeffs, batch)

    def compute_from_coeffs(
        self,
        values: torch.Tensor,
        coeffs: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute a linear combination of values and coefficients.

        This is a helper to compute `sum(coeffs * values)`.

        Parameters
        ----------
        values : torch.Tensor
            The pre-evaluated values, e.g., basis functions or their derivatives.
            Shape (N, num_components).
        coeffs : torch.Tensor
            The coefficients. Shape (M, num_components) or (N, num_components).
        batch : torch.Tensor, optional
            Mapping from N points to M coefficient sets. Defaults to None.

        Returns
        -------
        torch.Tensor
            The result of the linear combination, shape (N, 1).
        """

        if batch is None:
            return (coeffs * values).sum(dim=-1).unsqueeze(-1)
        else:
            return (coeffs[batch] * values).sum(dim=-1).unsqueeze(-1)

    def evaluate_derivatives(
        self, xy_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate all first and second order partial derivatives of the basis.

        Parameters
        ----------
        xy_data : torch.Tensor
            Input (u,v) coordinates of shape (..., 2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple of tensors: (dx, dy, dxdy, dxdx, dydy), each containing
            the evaluations of the corresponding partial derivative for every
            basis function.
        """

        dx, dy = self.gradient(xy_data).unbind(dim=-2)
        dxdx, dxdy, _, dydy = (
            self.hessian(xy_data).flatten(start_dim=-3, end_dim=-2).unbind(dim=-2)
        )

        return dx, dy, dxdy, dxdx, dydy

    def derivatives_from_coeffs(
        self,
        xy_data: torch.Tensor,
        coeffs: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute derivatives of a function defined by coefficients.

        The function is f(u,v) = sum(coeffs * L_{i,j}(u,v)). This method computes
        df/du, df/dv, d2f/dudv, d2f/du2, and d2f/dv2.

        Parameters
        ----------
        xy_data : torch.Tensor
            Input (u,v) coordinates of shape (N, 2).
        coeffs : torch.Tensor
            Coefficients for the Legendre basis.
        batch : torch.Tensor, optional
            Mapping from points to coefficient sets. Defaults to None.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 5) containing the five derivative values for
            each point.
        """

        dx, dy, dxdy, dxdx, dydy = self.evaluate_derivatives(xy_data)
        derivatives = torch.cat(
            (
                self.compute_from_coeffs(dx, coeffs, batch),
                self.compute_from_coeffs(dy, coeffs, batch),
                self.compute_from_coeffs(dxdy, coeffs, batch),
                self.compute_from_coeffs(dxdx, coeffs, batch),
                self.compute_from_coeffs(dydy, coeffs, batch),
            ),
            dim=-1,
        )
        return derivatives
