import math

import torch
import torch.nn.functional as F

from utils.func import transform


class SphericalRFFFRROT:
    """Random-Fourier fiber-robust loss for spherical tree projections."""

    def __init__(
        self,
        ntrees=200,
        nlines=5,
        rho=1.0,
        fiber_tau=1.0,
        num_frequencies=1,
        rff_sigma=1.0,
        delta=2.0,
        p=1,
        p_agg=2.0,
        device="cuda",
        type="normal",
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
            raise ValueError("RFF-FR-ROT uses tree-Wasserstein-1 on each tree")
        if p_agg < 1.0:
            raise ValueError("p_agg must be at least 1")
        if type not in ("normal", "generalized"):
            raise ValueError("type must be either 'normal' or 'generalized'")

        self.ntrees = ntrees
        self.nlines = nlines
        self.rho = float(rho)
        self.fiber_tau = float(fiber_tau)
        self.num_frequencies = int(num_frequencies)
        self.rff_sigma = float(rff_sigma)
        self.delta = delta
        self.p_agg = p_agg
        self.device = device
        self.type = type
        self.eps = 1e-6

    def __call__(self, X, Y):
        if self.type == "generalized":
            X = transform(X)
            Y = transform(Y)
        X = X.to(self.device)
        Y = Y.to(self.device)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be two-dimensional tensors")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same ambient dimension")

        root, meridians = self.generate_spherical_tree_frames(
            X.shape[1], X.dtype
        )
        omega = torch.randn(
            self.ntrees,
            self.nlines,
            self.num_frequencies,
            X.shape[1],
            device=X.device,
            dtype=X.dtype,
        ) / self.rff_sigma
        coord_X, mass_X, feature_X = self.project(X, root, meridians, omega)
        coord_Y, mass_Y, feature_Y = self.project(Y, root, meridians, omega)

        coordinates = torch.cat((coord_X, coord_Y), dim=-1)
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

    def project(self, samples, root, meridians, omega):
        num_samples = samples.shape[0]
        root_cosine = (root @ samples.T).squeeze(1)
        coordinates = torch.acos(
            torch.clamp(root_cosine, -1.0 + self.eps, 1.0 - self.eps)
        )

        tangent = samples.unsqueeze(0) - root_cosine.unsqueeze(-1) * root
        tangent = F.normalize(tangent, p=2, dim=-1, eps=1e-8)
        direction_cosine = meridians @ tangent.transpose(1, 2)
        angular_residual = torch.acos(
            torch.clamp(direction_cosine, -1.0 + self.eps, 1.0 - self.eps)
        )
        distances = angular_residual * torch.sin(coordinates).unsqueeze(1)
        masses = torch.softmax(-self.delta * distances, dim=1) / num_samples

        sin_coordinate = torch.sin(coordinates)
        # For z on the meridian at the same polar coordinate,
        # <x,z> = cos(alpha)^2 + sin(alpha)^2 <tangent(x), meridian>.
        point_dot_projection = (
            root_cosine.square().unsqueeze(1)
            + sin_coordinate.square().unsqueeze(1) * direction_cosine
        )
        residual_sq_norm = (2.0 - 2.0 * point_dot_projection).clamp_min(0.0)
        norm_eps = 1e-12
        residual_norm = torch.sqrt(residual_sq_norm + norm_eps) - math.sqrt(
            norm_eps
        )
        radial_gate = residual_norm / torch.sqrt(
            self.fiber_tau**2 + residual_norm.square()
        )

        omega_dot_x = torch.einsum("tlkd,nd->tlkn", omega, samples)
        omega_dot_root = torch.einsum("tlkd,tqd->tlkq", omega, root).squeeze(-1)
        omega_dot_meridian = torch.einsum("tlkd,tld->tlk", omega, meridians)
        phase = omega_dot_x - (
            omega_dot_root.unsqueeze(-1) * root_cosine[:, None, None, :]
            + omega_dot_meridian.unsqueeze(-1)
            * sin_coordinate[:, None, None, :]
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
        return coordinates, masses, features

    def tree_cost(self, signed_mass, signed_feature_mass, coordinates):
        coord_sorted, point_indices = torch.sort(coordinates, dim=-1)
        indices = point_indices.unsqueeze(1).expand(-1, signed_mass.shape[1], -1)
        mass_sorted = torch.gather(signed_mass, 2, indices)
        feature_indices = indices.unsqueeze(-1).expand(
            -1, -1, -1, signed_feature_mass.shape[-1]
        )
        feature_mass_sorted = torch.gather(
            signed_feature_mass, 2, feature_indices
        )
        mass_suffix = torch.flip(
            torch.cumsum(torch.flip(mass_sorted, dims=(2,)), dim=2), dims=(2,)
        )
        feature_suffix = torch.flip(
            torch.cumsum(torch.flip(feature_mass_sorted, dims=(2,)), dim=2),
            dims=(2,),
        )
        edge_discrepancy = torch.maximum(
            mass_suffix.abs(),
            self.rho * torch.linalg.vector_norm(feature_suffix, dim=-1),
        )
        edge_length = torch.diff(
            coord_sorted,
            prepend=torch.zeros(
                (coord_sorted.shape[0], 1),
                device=coord_sorted.device,
                dtype=coord_sorted.dtype,
            ),
            dim=-1,
        ).unsqueeze(1)
        return torch.sum(edge_length * edge_discrepancy, dim=(-1, -2))

    def generate_spherical_tree_frames(self, dimension, dtype):
        root = F.normalize(
            torch.randn(
                self.ntrees, 1, dimension, device=self.device, dtype=dtype
            ),
            p=2,
            dim=-1,
        )
        meridians = torch.randn(
            self.ntrees,
            self.nlines,
            dimension,
            device=self.device,
            dtype=dtype,
        )
        meridians = meridians - (meridians @ root.transpose(1, 2)) @ root
        meridians = F.normalize(meridians, p=2, dim=-1)
        return root, meridians


def rff_fr_rot(
    X,
    Y,
    ntrees=250,
    nlines=4,
    rho=1.0,
    fiber_tau=1.0,
    num_frequencies=1,
    rff_sigma=1.0,
    delta=2.0,
    p=1,
    p_agg=2.0,
    device="cuda",
    type="normal",
):
    loss = SphericalRFFFRROT(
        ntrees=ntrees,
        nlines=nlines,
        rho=rho,
        fiber_tau=fiber_tau,
        num_frequencies=num_frequencies,
        rff_sigma=rff_sigma,
        delta=delta,
        p=p,
        p_agg=p_agg,
        device=device,
        type=type,
    )
    return loss(X, Y)


def rff_fr_rot_unif(X, **kwargs):
    Y = F.normalize(torch.randn_like(X), p=2, dim=-1)
    return rff_fr_rot(X, Y, **kwargs)
