"""Current-PyTorch ports of the learned DK-for-TST baselines.

The architectures and objectives follow the ICML 2020 reference code while
removing Variable, implicit Softmax dimensions, .cuda() assumptions, and old
iterator APIs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tst_statistics import TestResult, finish_test, _quadratic_group_stat


class MLPFeature(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        width = 50 if dimension == 2 else 3 * dimension
        self.net = nn.Sequential(
            nn.Linear(dimension, width), nn.Softplus(),
            nn.Linear(width, width), nn.Softplus(),
            nn.Linear(width, width), nn.Softplus(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class ConvBackbone(nn.Module):
    def __init__(self, channels: int, image_size: int, output: int):
        super().__init__()
        layers = []
        for index, (cin, cout) in enumerate(zip((channels, 16, 32, 64), (16, 32, 64, 128))):
            layers.extend((nn.Conv2d(cin, cout, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)))
            if index:
                # The reference calls BatchNorm2d(cout, 0.8), where 0.8 is
                # PyTorch's second positional argument (`eps`).
                layers.append(nn.BatchNorm2d(cout, eps=0.8))
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(128 * (image_size // 16) ** 2, output)

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


def make_feature(sample: torch.Tensor, output: int | None = None):
    if sample.ndim == 4:
        return ConvBackbone(sample.shape[1], sample.shape[-1], output or (100 if sample.shape[1] == 1 else 300))
    return MLPFeature(sample[0].numel())


def pairwise_squared(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y).square().clamp_min(0)


def mmd_mean_variance(kernel: torch.Tensor, n_x: int):
    kx, ky = kernel[:n_x, :n_x], kernel[n_x:, n_x:]
    kxy = kernel[:n_x, n_x:]
    n = n_x
    mmd2 = (
        (kx.sum() - kx.diagonal().sum()) / (n * (n - 1))
        + (ky.sum() - ky.diagonal().sum()) / (n * (n - 1))
        - 2.0 * (kxy.sum() - kxy.diagonal().sum()) / (n * (n - 1))
    )
    hh = kx + ky - kxy - kxy.T
    v1 = torch.dot(hh.sum(1) / n, hh.sum(1) / n) / n
    v2 = hh.sum() / (n * n)
    return mmd2, (4.0 * (v1 - v2.square())).clamp_min(1e-8)


def gaussian_kernel(z: torch.Tensor, bandwidth_squared: torch.Tensor):
    return torch.exp(-pairwise_squared(z.flatten(1), z.flatten(1)) / bandwidth_squared.clamp_min(1e-8))


def train_mmd_o(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float):
    z = torch.cat((x, y)).flatten(1)
    initial = torch.pdist(z).square().median().clamp_min(1e-6)
    distances = pairwise_squared(z, z)
    log_bandwidth = nn.Parameter(initial.log())
    optimizer = torch.optim.Adam((log_bandwidth,), lr=lr)
    for _ in range(epochs):
        kernel = torch.exp(-distances / log_bandwidth.exp().clamp_min(1e-8))
        mmd2, variance = mmd_mean_variance(kernel, len(x))
        loss = -mmd2 / variance.sqrt()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # mmd_test uses exp(-d^2 / (2 sigma^2)).
    return float((log_bandwidth.exp().detach() / 2.0).sqrt())


def train_c2st(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, batch_size: int):
    feature = make_feature(x).to(x.device)
    with torch.no_grad():
        output_dimension = feature(x[:1]).shape[1]
    head = nn.Linear(output_dimension, 2).to(x.device)
    model = nn.Sequential(feature, nn.ReLU(), head) if x.ndim == 4 else nn.Sequential(feature, head)
    z = torch.cat((x, y))
    labels = torch.cat((torch.zeros(len(x)), torch.ones(len(y)))).long().to(x.device)
    loader = DataLoader(TensorDataset(z, labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch, target in loader:
            loss = criterion(model(batch), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def score_permutation_test(scores: torch.Tensor, labels: torch.Tensor, alpha: float):
    values = []
    for group in labels:
        values.append((scores[group].mean() - scores[~group].mean()).abs())
    return finish_test(torch.stack(values), alpha)


@torch.no_grad()
def c2st_test(model, x, y, labels, alpha: float, hard: bool):
    probabilities = model(torch.cat((x, y))).softmax(1)[:, 0]
    scores = (probabilities >= 0.5).to(probabilities.dtype) if hard else probabilities
    return score_permutation_test(scores, labels, alpha)


class DeepMMDState(nn.Module):
    def __init__(self, sample: torch.Tensor):
        super().__init__()
        self.feature = make_feature(sample)
        original_scale = (2.0 * 32.0 * 32.0) ** 0.5 if sample.ndim == 4 else float(sample[0].numel()) ** 0.5
        self.log_sigma = nn.Parameter(sample.new_tensor(original_scale).log())
        self.log_sigma0 = nn.Parameter(sample.new_tensor(0.005**0.5).log())
        self.epsilon_logit = nn.Parameter(sample.new_tensor(-20.0))

    def kernel(self, z):
        features = self.feature(z)
        feature_distance = pairwise_squared(features, features)
        original_distance = pairwise_squared(z.flatten(1), z.flatten(1))
        epsilon = self.epsilon_logit.sigmoid()
        sigma, sigma0 = self.log_sigma.exp().square(), self.log_sigma0.exp().square()
        return (1 - epsilon) * torch.exp(-feature_distance / sigma0 - original_distance / sigma) + epsilon * torch.exp(-original_distance / sigma)


def train_deep_mmd(x, y, epochs: int, lr: float, batch_size: int):
    state = DeepMMDState(x).to(x.device)
    optimizer = torch.optim.Adam(state.parameters(), lr=lr)
    for _ in range(epochs):
        order_x = torch.randperm(len(x), device=x.device)
        order_y = torch.randperm(len(y), device=y.device)
        for start in range(0, len(x), batch_size):
            xb = x[order_x[start:start + batch_size]]
            yb = y[order_y[start:start + batch_size]]
            if len(xb) < 2:
                continue
            kernel = state.kernel(torch.cat((xb, yb)))
            mmd2, variance = mmd_mean_variance(kernel, len(xb))
            loss = -mmd2 / variance.sqrt()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    state.eval()
    return state


@torch.no_grad()
def deep_mmd_test(state, x, y, labels, alpha: float):
    kernel = state.kernel(torch.cat((x, y)))
    return finish_test(_quadratic_group_stat(kernel, labels, "mmd"), alpha)


def train_freqopttest(x, y, method: str, alpha: float, locations: int = 5, seed: int = 15):
    """Train the original ME or SCF baseline through its maintained package."""
    try:
        import freqopttest.data as fot_data
        import freqopttest.tst as fot_tst
    except ImportError as error:
        raise ImportError(
            "ME/SCF require `pip install git+https://github.com/wittawatj/interpretable-test`."
        ) from error
    data = fot_data.TSTData(x.flatten(1).detach().cpu().numpy(), y.flatten(1).detach().cpu().numpy())
    if method == "me":
        options = dict(n_test_locs=locations, max_iter=300, locs_step_size=1.0, gwidth_step_size=0.1, tol_fun=1e-4, seed=seed + 5)
        test_locations, width, _ = fot_tst.MeanEmbeddingTest.optimize_locs_width(data, alpha, **options)
        return method, test_locations, width
    options = dict(n_test_freqs=locations, seed=seed, max_iter=300, batch_proportion=1.0, freqs_step_size=0.1, gwidth_step_size=0.01, tol_fun=1e-4)
    test_frequencies, width, _ = fot_tst.SmoothCFTest.optimize_freqs_width(data, alpha, **options)
    return method, test_frequencies, width


def freqopttest_test(state, x, y, alpha: float):
    import freqopttest.data as fot_data
    import freqopttest.tst as fot_tst

    method, locations, width = state
    data = fot_data.TSTData(x.flatten(1).detach().cpu().numpy(), y.flatten(1).detach().cpu().numpy())
    test = fot_tst.MeanEmbeddingTest(locations, width, alpha) if method == "me" else fot_tst.SmoothCFTest(locations, width, alpha=alpha)
    result = test.perform_test(data)
    return TestResult(
        float(result.get("test_stat", float("nan"))),
        float(result.get("pvalue", float("nan"))),
        bool(result["h0_rejected"]),
    )
