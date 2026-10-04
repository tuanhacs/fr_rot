"""Datasets used by Liu et al. (ICML 2020) two-sample experiments.

The Blob and HDGM generators intentionally preserve the distributions and seed
schedule of https://github.com/fengliu90/DK-for-TST.  Image loaders expose the
same real/fake MNIST and CIFAR-10/CIFAR-10.1 pools without retaining the old
PyTorch 1.1 training code.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch
import numpy as np


def blob_covariances() -> np.ndarray:
    base = np.array([[0.03, 0.0], [0.0, 0.03]])
    covariances = np.repeat(base[None], 9, axis=0)
    for i in range(9):
        correlation = -0.02 - 0.002 * i if i < 4 else (0.02 + 0.002 * (i - 5) if i > 4 else 0.0)
        covariances[i, 0, 1] = correlation
        covariances[i, 1, 0] = correlation
    return covariances


def sample_blob(n_per_mode: int, seed: int, null: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Blob-S under H0 and Blob-D under H1, following DK-for-TST."""
    generator = torch.Generator().manual_seed(seed)
    n = 9 * n_per_mode
    locations_x = torch.stack((torch.randint(3, (n,), generator=generator), torch.randint(3, (n,), generator=generator)), 1).float()
    locations_y = torch.stack((torch.randint(3, (n,), generator=generator), torch.randint(3, (n,), generator=generator)), 1).float()
    if null:
        return torch.randn(n, 2, generator=generator) + locations_x, torch.randn(n, 2, generator=generator) + locations_y

    x = math.sqrt(0.03) * torch.randn(n, 2, generator=generator) + locations_x
    raw_y = torch.randn(n, 2, generator=generator)
    rows, cols = locations_y[:, 0].long(), locations_y[:, 1].long()
    y = torch.empty_like(raw_y)
    covariances = torch.from_numpy(blob_covariances()).float()
    for row in range(3):
        for col in range(3):
            mask = (rows == row) & (cols == col)
            transform = torch.linalg.cholesky(covariances[3 * row + col])
            y[mask] = raw_y[mask] @ transform.T + torch.tensor([row, col])
    return x, y


def sample_hdgm(n_per_mode: int, dimension: int, seed: int, null: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two-mode high-dimensional Gaussian mixture from DK-for-TST."""
    if dimension < 2:
        raise ValueError("HDGM requires dimension >= 2")
    means = torch.zeros(2, dimension)
    means[1] = 0.5
    covariance_p = torch.eye(dimension)
    covariance_q = [torch.eye(dimension), torch.eye(dimension)]
    covariance_q[0][0, 1] = covariance_q[0][1, 0] = 0.5
    covariance_q[1][0, 1] = covariance_q[1][1, 0] = -0.5
    x_parts, y_parts = [], []
    for mode in range(2):
        x_rng = torch.Generator().manual_seed(1102 * seed + mode + n_per_mode)
        y_rng = torch.Generator().manual_seed(819 * seed + 1 + mode + n_per_mode)
        x_parts.append(torch.randn(n_per_mode, dimension, generator=x_rng) + means[mode])
        y_cov = covariance_p if null else covariance_q[mode]
        chol = torch.linalg.cholesky(y_cov)
        y_parts.append(torch.randn(n_per_mode, dimension, generator=y_rng) @ chol.T + means[mode])
    return torch.cat(x_parts), torch.cat(y_parts)


def _unwrap_fake_mnist(value) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 3:
        value = value[:, None]
    if value.ndim != 4:
        raise ValueError(f"Expected fake MNIST with 3 or 4 dimensions, got {value.shape}")
    return value.astype("float32")


def load_mnist_pools(data_dir: str | Path, fake_path: str | Path, image_size: int = 32):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    real = torch.stack([train[i][0] for i in range(4000)])
    with Path(fake_path).open("rb") as handle:
        fake = torch.from_numpy(_unwrap_fake_mnist(pickle.load(handle)))
    if len(fake) < 4000:
        raise ValueError("The DK-for-TST protocol needs at least 4000 fake MNIST images")
    if fake.shape[-2:] != (image_size, image_size):
        fake = torch.nn.functional.interpolate(fake, (image_size, image_size), mode="bilinear", align_corners=False)
    return real.float(), fake[:4000].float()


def load_cifar_pools(data_dir: str | Path, cifar101_path: str | Path, image_size: int = 64):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cifar = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform)
    real = torch.stack([cifar[i][0] for i in range(len(cifar))])
    raw = np.load(cifar101_path)
    if raw.ndim != 4:
        raise ValueError(f"Expected CIFAR-10.1 NHWC array, got {raw.shape}")
    # ToPILImage reproduces the reference preprocessing for uint8 data.
    to_pil = transforms.ToPILImage()
    shifted = torch.stack([transform(to_pil(image)) for image in raw])
    return real.float(), shifted.float()


def split_image_pools(real: torch.Tensor, shifted: torch.Tensor, n_train: int, seed: int):
    """DK-for-TST-style disjoint train/test pools."""
    if n_train >= min(len(real), len(shifted)):
        raise ValueError("n_train must be smaller than both image pools")
    rng_real = np.random.RandomState(1102 * (seed + 10) + n_train)
    rng_shifted = np.random.RandomState(819 * (seed + 9) + n_train)
    real_train = rng_real.choice(len(real), n_train, replace=False)
    shifted_train = rng_shifted.choice(len(shifted), n_train, replace=False)
    real_test = np.delete(np.arange(len(real)), real_train)
    shifted_test = np.delete(np.arange(len(shifted)), shifted_train)
    return real[real_train], shifted[shifted_train], real[real_test], shifted[shifted_test]


def sample_image_pair(real_pool: torch.Tensor, shifted_pool: torch.Tensor, n: int, seed: int):
    if n > min(len(real_pool), len(shifted_pool)):
        raise ValueError("sample size exceeds an image test pool")
    rng_x = np.random.RandomState(1102 * (seed + 1) + n)
    rng_y = np.random.RandomState(819 * (seed + 3) + n)
    ix = rng_x.choice(len(real_pool), n, replace=False)
    iy = rng_y.choice(len(shifted_pool), n, replace=False)
    return real_pool[ix], shifted_pool[iy]
