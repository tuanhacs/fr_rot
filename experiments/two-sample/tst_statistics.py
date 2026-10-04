"""Modern permutation tests for tree discrepancies and standard baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TestResult:
    statistic: float
    p_value: float
    reject: bool


def permutation_labels(total: int, n_x: int, permutations: int, generator: torch.Generator, device):
    labels = torch.zeros(permutations + 1, total, dtype=torch.bool, device=device)
    labels[0, :n_x] = True
    for row in range(1, permutations + 1):
        labels[row, torch.randperm(total, generator=generator, device=device)[:n_x]] = True
    return labels


def finish_test(values: torch.Tensor, alpha: float) -> TestResult:
    observed, null = values[0], values[1:]
    p_value = (1.0 + (null >= observed).sum().item()) / (len(null) + 1.0)
    return TestResult(float(observed), float(p_value), bool(p_value <= alpha))


def _quadratic_group_stat(matrix: torch.Tensor, labels: torch.Tensor, kind: str, chunk: int = 128):
    diagonal = torch.diagonal(matrix)
    n_x = int(labels[0].sum())
    n_y = labels.shape[1] - n_x
    outputs = []
    for start in range(0, len(labels), chunk):
        x = labels[start:start + chunk].to(matrix.dtype)
        y = 1.0 - x
        xx = torch.einsum("pi,ij,pj->p", x, matrix, x)
        yy = torch.einsum("pi,ij,pj->p", y, matrix, y)
        xy = torch.einsum("pi,ij,pj->p", x, matrix, y)
        if kind == "mmd":
            xx = (xx - x @ diagonal) / (n_x * (n_x - 1))
            yy = (yy - y @ diagonal) / (n_y * (n_y - 1))
            outputs.append(xx + yy - 2.0 * xy / (n_x * n_y))
        elif kind == "energy":
            xx = (xx - x @ diagonal) / (n_x * (n_x - 1))
            yy = (yy - y @ diagonal) / (n_y * (n_y - 1))
            outputs.append(2.0 * xy / (n_x * n_y) - xx - yy)
        else:
            raise ValueError(kind)
    return torch.cat(outputs)


def median_bandwidth(z: torch.Tensor, max_points: int = 2000) -> torch.Tensor:
    if len(z) > max_points:
        z = z[:max_points]
    distances = torch.pdist(z)
    positive = distances[distances > 0]
    return positive.median().clamp_min(torch.finfo(z.dtype).eps)


def mmd_test(z: torch.Tensor, labels: torch.Tensor, alpha: float, bandwidth: float | None = None):
    bandwidth_t = median_bandwidth(z) if bandwidth is None else z.new_tensor(bandwidth)
    sq_dist = torch.cdist(z, z).square()
    kernel = torch.exp(-sq_dist / (2.0 * bandwidth_t.square()))
    return finish_test(_quadratic_group_stat(kernel, labels, "mmd"), alpha)


def energy_test(z: torch.Tensor, labels: torch.Tensor, alpha: float):
    distances = torch.cdist(z, z)
    return finish_test(_quadratic_group_stat(distances, labels, "energy"), alpha)


class TreePermutationStatistic:
    """Cache all label-independent work and batch the label permutations."""

    def __init__(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        intercept: torch.Tensor,
        method: str,
        delta: float = 2.0,
        rho: float = 1.0,
        fiber_tau: float = 1.0,
        p_agg: float = 1.0,
        num_frequencies: int = 4,
        rff_sigma: float = 1.0,
        omega: torch.Tensor | None = None,
        fusion: str = "max",
    ):
        if method not in {"dbtsw", "fr_rot", "rff_fr_rot"}:
            raise ValueError(method)
        if fusion not in {"max", "add"}:
            raise ValueError("fusion must be max or add")
        self.method, self.rho, self.p_agg, self.fusion = method, rho, p_agg, fusion
        centre = intercept[:, 0]
        # Avoid [tree, point, dimension] storage. This matters for raw images.
        coordinates = torch.einsum("tld,nd->tln", theta, z) - torch.einsum("tld,td->tl", theta, centre).unsqueeze(-1)
        translated_sq = (
            z.square().sum(-1).unsqueeze(0)
            - 2.0 * torch.einsum("nd,td->tn", z, centre)
            + centre.square().sum(-1, keepdim=True)
        )
        residual_sq = (translated_sq.unsqueeze(1) - coordinates.square()).clamp_min(0)
        branch_prob = torch.softmax(-delta * torch.sqrt(residual_sq + 1e-12), dim=1)

        features = None
        if method == "fr_rot":
            projected = torch.einsum("tln,tld->tlnd", coordinates, theta)
            residual = z[None, None] - centre[:, None, None] - projected
            features = residual / torch.sqrt(fiber_tau**2 + residual.square().sum(-1, keepdim=True))
        elif method == "rff_fr_rot":
            if omega is None:
                omega = torch.randn(theta.shape[0], num_frequencies, z.shape[1], device=z.device, dtype=z.dtype) / rff_sigma
            residual_norm = torch.sqrt(residual_sq + 1e-12) - 1e-6
            gate = residual_norm / torch.sqrt(fiber_tau**2 + residual_norm.square())
            omega_x = torch.einsum("tkd,nd->tkn", omega, z) - torch.einsum("tkd,td->tk", omega, centre).unsqueeze(-1)
            omega_theta = torch.einsum("tkd,tld->tlk", omega, theta)
            phase = omega_x.unsqueeze(1) - omega_theta.unsqueeze(-1) * coordinates.unsqueeze(2)
            fourier = torch.cat((torch.cos(phase).transpose(2, 3), torch.sin(phase).transpose(2, 3)), -1)
            features = gate.unsqueeze(-1) * torch.cat((torch.ones_like(gate).unsqueeze(-1), fourier), -1)
            features = features / math.sqrt(num_frequencies + 1)

        coord_sorted, order = coordinates.sort(dim=2)
        self.branch_prob = branch_prob.gather(2, order)
        self.features = None if features is None else features.gather(2, order.unsqueeze(-1).expand(*order.shape, features.shape[-1]))
        right = coord_sorted > 0
        self.right = right
        root = torch.zeros(*coord_sorted.shape[:2], 1, device=z.device, dtype=z.dtype)
        root_index = torch.searchsorted(coord_sorted.contiguous(), root)
        with_root = torch.zeros(*coord_sorted.shape[:2], coord_sorted.shape[2] + 1, device=z.device, dtype=z.dtype)
        non_root = torch.ones_like(with_root, dtype=torch.bool)
        non_root.scatter_(2, root_index, False)
        with_root[non_root] = coord_sorted.reshape(-1)
        self.edge_length = with_root[..., 1:] - with_root[..., :-1]
        self.order = order

    def __call__(self, labels: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        n_x = int(labels[0].sum())
        n_y = labels.shape[1] - n_x
        outputs = []
        for start in range(0, len(labels), chunk):
            group = labels[start:start + chunk]
            coefficient = torch.where(group, 1.0 / n_x, -1.0 / n_y).to(self.branch_prob.dtype)
            sorted_coefficient = coefficient[:, None, None, :].expand(-1, *self.order.shape).gather(3, self.order.unsqueeze(0).expand(len(group), -1, -1, -1))
            mass = self.branch_prob.unsqueeze(0) * sorted_coefficient
            prefix = mass.cumsum(3)
            suffix = mass + mass.sum(3, keepdim=True) - prefix
            subtree_mass = torch.where(self.right.unsqueeze(0), suffix, prefix)
            edge_value = subtree_mass.abs()
            if self.features is not None:
                feature_mass = mass.unsqueeze(-1) * self.features.unsqueeze(0)
                feature_prefix = feature_mass.cumsum(3)
                feature_suffix = feature_mass + feature_mass.sum(3, keepdim=True) - feature_prefix
                subtree_feature = torch.where(self.right[None, ..., None], feature_suffix, feature_prefix)
                fiber = self.rho * torch.linalg.vector_norm(subtree_feature, dim=-1)
                edge_value = torch.maximum(edge_value, fiber) if self.fusion == "max" else edge_value + fiber
            tree_cost = (self.edge_length.unsqueeze(0) * edge_value).sum(dim=(-1, -2))
            outputs.append(tree_cost.pow(self.p_agg).mean(1).pow(1.0 / self.p_agg))
        return torch.cat(outputs)


def generate_tree_parameters(z: torch.Tensor, ntrees: int, nlines: int, generator: torch.Generator, root_std: float = 0.0):
    d = z.shape[1]
    theta = torch.randn(ntrees, nlines, d, generator=generator, device=z.device, dtype=z.dtype)
    theta = theta / theta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    centre = z.mean(0).view(1, 1, d)
    scale = z.std(0).mean().clamp_min(1e-12)
    intercept = centre.expand(ntrees, 1, d).clone()
    if root_std:
        intercept += root_std * scale * torch.randn(ntrees, 1, d, generator=generator, device=z.device, dtype=z.dtype)
    return theta, intercept


def tree_test(z: torch.Tensor, labels: torch.Tensor, alpha: float, seed: int, permutation_chunk: int = 16, **kwargs):
    generator = torch.Generator(device=z.device).manual_seed(seed)
    theta, intercept = generate_tree_parameters(
        z, kwargs.pop("ntrees"), kwargs.pop("nlines"), generator, kwargs.pop("root_std", 0.0)
    )
    if kwargs.get("method") == "rff_fr_rot":
        k = kwargs.get("num_frequencies", 4)
        kwargs["omega"] = torch.randn(theta.shape[0], k, z.shape[1], generator=generator, device=z.device, dtype=z.dtype) / kwargs.get("rff_sigma", 1.0)
    statistic = TreePermutationStatistic(z, theta, intercept, **kwargs)
    return finish_test(statistic(labels, chunk=permutation_chunk), alpha)
