import torch
import torch.nn.functional as F

from utils.func import transform


class SphericalFRROT:
    """Fiber-robust tree-Wasserstein loss for spherical concurrent-line trees.

    A spherical tree keeps only the polar coordinate of a point and softly
    assigns its mass to sampled meridians.  For a point ``x`` assigned to a
    meridian, the fiber feature is the normalized chord residual between
    ``x`` and the point on that meridian with the same polar coordinate.

    On every edge the nominal subtree imbalance ``m`` is replaced by

        max(abs(m), rho * ||b||_2),

    where ``b`` is the signed cumulative fiber feature.  This is the closed
    form of the binary fiber uncertainty set; edge lengths remain fixed.
    """

    def __init__(
        self,
        ntrees=200,
        nlines=5,
        rho=1.0,
        fiber_tau=1.0,
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
        if p != 1:
            raise ValueError("FR-ROT uses tree-Wasserstein-1 on each spherical tree")
        if p_agg < 1.0:
            raise ValueError("p_agg must be at least 1")
        if type not in ("normal", "generalized"):
            raise ValueError("type must be either 'normal' or 'generalized'")

        self.ntrees = ntrees
        self.nlines = nlines
        self.rho = float(rho)
        self.fiber_tau = float(fiber_tau)
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

        root, meridians = self.generate_spherical_tree_frames(X.shape[1])
        coord_X, mass_X, feature_X = self.project(X, root, meridians)
        coord_Y, mass_Y, feature_Y = self.project(Y, root, meridians)

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

    def project(self, samples, root, meridians):
        """Return polar coordinates, split masses, and spherical fibers."""
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

        # Point on each sampled meridian at the same polar coordinate.
        projected = (
            torch.cos(coordinates).unsqueeze(1).unsqueeze(-1) * root.unsqueeze(1)
            + torch.sin(coordinates).unsqueeze(1).unsqueeze(-1)
            * meridians.unsqueeze(2)
        )
        residual = samples.unsqueeze(0).unsqueeze(0) - projected
        residual_sq_norm = residual.square().sum(dim=-1, keepdim=True)
        features = residual / torch.sqrt(self.fiber_tau**2 + residual_sq_norm)

        return coordinates, masses, features

    def tree_cost(self, signed_mass, signed_feature_mass, coordinates):
        """Compute the fiber-robust cost for every sampled spherical tree."""
        coord_sorted, point_indices = torch.sort(coordinates, dim=-1)
        indices = point_indices.unsqueeze(1).expand(
            -1, signed_mass.shape[1], -1
        )
        mass_sorted = torch.gather(signed_mass, 2, indices)
        feature_indices = indices.unsqueeze(-1).expand(
            -1, -1, -1, signed_feature_mass.shape[-1]
        )
        feature_mass_sorted = torch.gather(
            signed_feature_mass, 2, feature_indices
        )

        # Every branch starts at polar coordinate zero, so its edge subtree is
        # the suffix of the sorted points assigned to that branch.
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

    def generate_spherical_tree_frames(self, dimension):
        root = torch.randn(
            self.ntrees, 1, dimension, device=self.device
        )
        root = F.normalize(root, p=2, dim=-1)

        meridians = torch.randn(
            self.ntrees, self.nlines, dimension, device=self.device
        )
        meridians = meridians - (meridians @ root.transpose(1, 2)) @ root
        meridians = F.normalize(meridians, p=2, dim=-1)
        return root, meridians


def fr_rot(
    X,
    Y,
    ntrees=250,
    nlines=4,
    rho=1.0,
    fiber_tau=1.0,
    delta=2.0,
    p=1,
    p_agg=2.0,
    device="cuda",
    type="normal",
):
    loss = SphericalFRROT(
        ntrees=ntrees,
        nlines=nlines,
        rho=rho,
        fiber_tau=fiber_tau,
        delta=delta,
        p=p,
        p_agg=p_agg,
        device=device,
        type=type,
    )
    return loss(X, Y)


def fr_rot_unif(X, **kwargs):
    Y = F.normalize(torch.randn_like(X), p=2, dim=-1)
    return fr_rot(X, Y, **kwargs)
