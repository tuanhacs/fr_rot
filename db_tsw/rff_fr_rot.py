import math

import torch


class RFFFRROTConcurrentLines:
    """Random-Fourier fiber-robust loss on concurrent-line trees.

    The residual ``r`` from a point to a tree line is represented by

        g_tau(||r||) / sqrt(K + 1)
        * [1, cos(w_1^T r), sin(w_1^T r), ..., cos(w_K^T r), sin(w_K^T r)],

    where ``w_k ~ N(0, rff_sigma^{-2} I)`` and
    ``g_tau(s) = s / sqrt(tau^2 + s^2)``.  Thus every feature has norm at
    most one and the same binary-refinement closed form as FR-ROT applies:

        max(abs(m), rho * ||b||_2).

    Only ``1 + 2K`` cumulative feature channels are stored, independently
    of the ambient dimension.  The full residual tensor is never formed.
    """

    def __init__(
        self,
        rho=1.0,
        fiber_tau=1.0,
        num_frequencies=1,
        rff_sigma=1.0,
        delta=2.0,
        mass_division="distance_based",
        p=1,
        p_agg=1,
        device="cuda",
    ):
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1] for valid binary refinements")
        if fiber_tau <= 0.0:
            raise ValueError("fiber_tau must be strictly positive")
        if int(num_frequencies) != num_frequencies or num_frequencies < 1:
            raise ValueError("num_frequencies must be a positive integer")
        if rff_sigma <= 0.0:
            raise ValueError("rff_sigma must be strictly positive")
        if p != 1:
            raise ValueError("RFF-FR-ROT is defined for tree-Wasserstein-1 only")
        if p_agg < 1:
            raise ValueError("p_agg must be at least 1")
        if mass_division not in ("uniform", "distance_based"):
            raise ValueError(
                "mass_division must be either 'uniform' or 'distance_based'"
            )

        self.rho = float(rho)
        self.fiber_tau = float(fiber_tau)
        self.num_frequencies = int(num_frequencies)
        self.rff_sigma = float(rff_sigma)
        self.delta = delta
        self.mass_division = mass_division
        self.p_agg = p_agg
        self.device = device

    def __call__(self, X, Y, theta, intercept):
        device = X.device
        Y = Y.to(device)
        theta = theta.to(device=device, dtype=X.dtype)
        intercept = intercept.to(device=device, dtype=X.dtype)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be two-dimensional tensors")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same ambient dimension")

        omega = torch.randn(
            *theta.shape[:2],
            self.num_frequencies,
            X.shape[1],
            device=device,
            dtype=X.dtype,
        ) / self.rff_sigma

        coordinate_X, mass_X, feature_X = self.project(
            X, theta, intercept, omega
        )
        coordinate_Y, mass_Y, feature_Y = self.project(
            Y, theta, intercept, omega
        )

        coordinates = torch.cat((coordinate_X, coordinate_Y), dim=2)
        signed_mass = torch.cat((mass_X, -mass_Y), dim=2)
        signed_feature_mass = torch.cat(
            (
                mass_X.unsqueeze(-1) * feature_X,
                -mass_Y.unsqueeze(-1) * feature_Y,
            ),
            dim=2,
        )
        costs = self.tree_cost(signed_mass, signed_feature_mass, coordinates)
        return costs.pow(self.p_agg).mean().pow(1.0 / self.p_agg)

    def project(self, samples, theta, intercept, omega):
        """Project points and compute RFF fibers without materializing residuals."""
        num_samples = samples.shape[0]
        num_trees, num_lines, _ = theta.shape

        translated = samples.unsqueeze(0) - intercept
        coordinates = torch.matmul(theta, translated.transpose(1, 2))

        # theta is unit length, hence ||r||^2 = ||x-c||^2 - <theta,x-c>^2.
        translated_sq_norm = translated.square().sum(dim=-1)
        residual_sq_norm = (
            translated_sq_norm.unsqueeze(1) - coordinates.square()
        ).clamp_min(0.0)
        # A direct sqrt has an infinite derivative at zero when the residual is
        # recovered from its squared norm.  This smooth zero-preserving norm
        # avoids NaNs for points lying exactly on a sampled line.
        norm_eps = 1e-12
        residual_norm = torch.sqrt(residual_sq_norm + norm_eps) - math.sqrt(
            norm_eps
        )
        radial_gate = residual_norm / torch.sqrt(
            self.fiber_tau**2 + residual_norm.square()
        )

        # w^T r = w^T(x-c) - <theta,x-c><w,theta>.
        omega_dot_translated = torch.einsum("tlkd,tnd->tlkn", omega, translated)
        omega_dot_theta = torch.einsum("tlkd,tld->tlk", omega, theta)
        phase = omega_dot_translated - (
            omega_dot_theta.unsqueeze(-1) * coordinates.unsqueeze(2)
        )
        fourier = torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)
        fourier = fourier.permute(0, 1, 3, 2, 4).flatten(start_dim=-2)
        features = torch.cat(
            (torch.ones_like(radial_gate).unsqueeze(-1), fourier), dim=-1
        )
        features = (
            radial_gate.unsqueeze(-1)
            * features
            / math.sqrt(self.num_frequencies + 1)
        )

        if self.mass_division == "uniform":
            masses = torch.full(
                (num_trees, num_lines, num_samples),
                1.0 / (num_samples * num_lines),
                device=samples.device,
                dtype=samples.dtype,
            )
        else:
            masses = torch.softmax(-self.delta * residual_norm, dim=1) / num_samples

        return coordinates, masses, features

    def tree_cost(self, signed_mass, signed_feature_mass, coordinates):
        coord_sorted, indices = torch.sort(coordinates, dim=2)
        mass_sorted = torch.gather(signed_mass, 2, indices)
        feature_indices = indices.unsqueeze(-1).expand(
            *indices.shape, signed_feature_mass.shape[-1]
        )
        feature_mass_sorted = torch.gather(
            signed_feature_mass, 2, feature_indices
        )

        mass_prefix = torch.cumsum(mass_sorted, dim=2)
        feature_prefix = torch.cumsum(feature_mass_sorted, dim=2)
        mass_suffix = mass_sorted + mass_sorted.sum(dim=2, keepdim=True) - mass_prefix
        feature_suffix = (
            feature_mass_sorted
            + feature_mass_sorted.sum(dim=2, keepdim=True)
            - feature_prefix
        )

        right = coord_sorted > 0
        subtree_mass = torch.where(right, mass_suffix, mass_prefix)
        subtree_feature = torch.where(
            right.unsqueeze(-1), feature_suffix, feature_prefix
        )
        edge_discrepancy = torch.maximum(
            subtree_mass.abs(),
            self.rho * torch.linalg.vector_norm(subtree_feature, dim=-1),
        )

        root = torch.zeros(
            (*coord_sorted.shape[:2], 1),
            device=coord_sorted.device,
            dtype=coord_sorted.dtype,
        )
        root_indices = torch.searchsorted(coord_sorted.contiguous(), root)
        coordinates_with_root = torch.zeros(
            (*coord_sorted.shape[:2], coord_sorted.shape[2] + 1),
            device=coord_sorted.device,
            dtype=coord_sorted.dtype,
        )
        non_root = torch.ones_like(coordinates_with_root, dtype=torch.bool)
        non_root.scatter_(2, root_indices, False)
        coordinates_with_root[non_root] = coord_sorted.reshape(-1)
        edge_length = coordinates_with_root[:, :, 1:] - coordinates_with_root[:, :, :-1]
        return torch.sum(edge_length * edge_discrepancy, dim=(-1, -2))


class RFFFRTW(RFFFRROTConcurrentLines):
    """Short alias for random-Fourier fiber-robust tree-Wasserstein."""
