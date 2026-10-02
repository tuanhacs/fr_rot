import torch


class FRROTConcurrentLines:
    """Fiber-robust tree-Wasserstein loss on concurrent-line trees.

    For every tree edge, the usual signed subtree mass ``m`` is augmented
    with the signed cumulative residual feature ``b``.  The edge discrepancy
    is

        max(abs(m), rho * ||b||_2).

    This is the closed form of the worst-case binary fiber refinement with
    linear, bounded residual features.  No edge-weight uncertainty is used.
    """

    def __init__(
        self,
        rho=1.0,
        fiber_tau=1.0,
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
        if p != 1:
            raise ValueError("FR-ROT is currently defined for tree-Wasserstein-1 only")
        if p_agg < 1:
            raise ValueError("p_agg must be at least 1")
        if mass_division not in ("uniform", "distance_based"):
            raise ValueError(
                "mass_division must be either 'uniform' or 'distance_based'"
            )

        self.rho = float(rho)
        self.fiber_tau = float(fiber_tau)
        self.delta = delta
        self.mass_division = mass_division
        self.p = p
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

        coordinate_X, mass_X, feature_X = self.project(X, theta, intercept)
        coordinate_Y, mass_Y, feature_Y = self.project(Y, theta, intercept)

        coordinates = torch.cat((coordinate_X, coordinate_Y), dim=2)
        signed_mass = torch.cat((mass_X, -mass_Y), dim=2)
        features = torch.cat((feature_X, feature_Y), dim=2)
        signed_feature_mass = signed_mass.unsqueeze(-1) * features

        tree_cost = self.tree_cost(
            signed_mass,
            signed_feature_mass,
            coordinates,
        )
        return tree_cost.pow(self.p_agg).mean().pow(1.0 / self.p_agg)

    def project(self, samples, theta, intercept):
        """Project samples and return coordinates, masses, and fiber features."""
        num_samples = samples.shape[0]
        num_trees, num_lines, _ = theta.shape

        # [num_trees, num_samples, ambient_dimension]
        translated = samples.unsqueeze(0) - intercept
        # [num_trees, num_lines, num_samples]
        coordinates = torch.matmul(theta, translated.transpose(1, 2))
        projected = torch.einsum("tln,tld->tlnd", coordinates, theta)
        residual = translated.unsqueeze(1) - projected

        residual_sq_norm = residual.square().sum(dim=-1, keepdim=True)
        features = residual / torch.sqrt(self.fiber_tau**2 + residual_sq_norm)

        if self.mass_division == "uniform":
            masses = torch.full(
                (num_trees, num_lines, num_samples),
                1.0 / (num_samples * num_lines),
                device=samples.device,
                dtype=samples.dtype,
            )
        else:
            distances = torch.sqrt(residual_sq_norm.squeeze(-1))
            masses = torch.softmax(-self.delta * distances, dim=1) / num_samples

        return coordinates, masses, features

    def tree_cost(self, signed_mass, signed_feature_mass, coordinates):
        """Compute one FR-ROT value per sampled tree."""
        coord_sorted, indices = torch.sort(coordinates, dim=2)
        mass_sorted = torch.gather(signed_mass, 2, indices)

        feature_indices = indices.unsqueeze(-1).expand(
            *indices.shape,
            signed_feature_mass.shape[-1],
        )
        feature_mass_sorted = torch.gather(
            signed_feature_mass,
            2,
            feature_indices,
        )

        mass_prefix = torch.cumsum(mass_sorted, dim=2)
        feature_prefix = torch.cumsum(feature_mass_sorted, dim=2)

        mass_suffix = (
            mass_sorted
            + mass_sorted.sum(dim=2, keepdim=True)
            - mass_prefix
        )
        feature_suffix = (
            feature_mass_sorted
            + feature_mass_sorted.sum(dim=2, keepdim=True)
            - feature_prefix
        )

        right = coord_sorted > 0
        subtree_mass = torch.where(right, mass_suffix, mass_prefix)
        subtree_feature = torch.where(
            right.unsqueeze(-1),
            feature_suffix,
            feature_prefix,
        )

        abs_mass = subtree_mass.abs()
        fiber_discrepancy = torch.linalg.vector_norm(subtree_feature, dim=-1)
        edge_discrepancy = torch.maximum(
            abs_mass,
            self.rho * fiber_discrepancy,
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
        edge_length = (
            coordinates_with_root[:, :, 1:]
            - coordinates_with_root[:, :, :-1]
        )

        # Zero-length intervals are not geometric tree edges and contribute zero.
        return torch.sum(edge_length * edge_discrepancy, dim=(-1, -2))


class FRTW(FRROTConcurrentLines):
    """Short alias for the fiber-only robust tree-Wasserstein loss."""

