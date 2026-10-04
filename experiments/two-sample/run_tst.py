"""Two-sample power experiments for fiber tree distances.

The default ``sow`` protocol follows Section 4.2 of Sliced
Orlicz-Wasserstein: 200 independent trials, 199 permutations, alpha=.05, and a
projection budget L=n/2. The earlier DK-for-TST reproduction remains available
through ``--protocol dk``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from datasets import (  # noqa: E402
    load_cifar_pools,
    load_mnist_digit_pools,
    load_mnist_pools,
    sample_blob,
    sample_hdgm,
    sample_image_pair,
    sample_mnist_mixture_pair,
    sample_rare_gaussian,
    split_image_pools,
)
from learned_baselines import (  # noqa: E402
    c2st_test,
    deep_mmd_test,
    freqopttest_test,
    train_c2st,
    train_deep_mmd,
    train_freqopttest,
    train_mmd_o,
)
from tst_statistics import (  # noqa: E402
    energy_test,
    mmd_test,
    permutation_labels,
    sliced_wasserstein_test,
    tree_test,
)

TREE_METHODS = {"dbtsw", "fr_rot", "rff_fr_rot"}
LEARNED_METHODS = {"mmd_o", "mmd_d", "c2st_s", "c2st_l", "me", "scf"}
ALL_METHODS = (
    "dbtsw", "fr_rot", "rff_fr_rot", "sw2", "mmd", "mmd_rbf", "mmd_laplace",
    "mmd_o", "mmd_d", "energy", "c2st_s", "c2st_l", "me", "scf",
)
SOW_DATASETS = {"rare_gaussian", "mnist", "cifar10"}
DK_DATASETS = {"blob", "hdgm", "mnist_fake", "cifar10"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=("sow", "dk"), default="sow")
    p.add_argument("--dataset", choices=tuple(sorted(SOW_DATASETS | DK_DATASETS)), required=True)
    p.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS,
        default=None,
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--permutations", type=int, default=None)
    p.add_argument("--trials", type=int, default=None, help="Independent trials in the SOW protocol")
    p.add_argument("--outer-trials", type=int, default=10, help="Outer trials in the legacy DK protocol")
    p.add_argument("--test-sets", type=int, default=100, help="Test sets per outer trial in the DK protocol")
    p.add_argument("--seed", type=int, default=1102)
    p.add_argument("--null", action="store_true", help="Estimate type-I error instead of power")
    p.add_argument("--n", type=int, default=None, help="Run one sample size per distribution")
    p.add_argument("--sample-sizes", nargs="+", type=int, default=None, help="Run a power curve; overrides --n")
    p.add_argument("--dimension", type=int, default=None)
    p.add_argument("--contamination", type=float, default=0.1)
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--fake-mnist-path", default=str(HERE / "data" / "Fake_MNIST_data_EP100_N10000.pckl"))
    p.add_argument("--cifar101-path", default=str(HERE / "data" / "cifar10.1_v4_data.npy"))
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument(
        "--ntrees", type=int, default=None,
        help="Override tree count; SOW otherwise uses ntrees*nlines ~= n/2",
    )
    p.add_argument("--nlines", type=int, default=4)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--fiber-tau", type=float, default=1.0)
    p.add_argument("--p-agg", type=float, default=None)
    p.add_argument("--num-frequencies", type=int, default=4)
    p.add_argument("--rff-sigma", type=float, default=1.0)
    p.add_argument("--fusion", choices=("max", "add"), default="max")
    p.add_argument("--root-std", type=float, default=0.0)
    p.add_argument("--permutation-chunk", type=int, default=16)
    p.add_argument("--train-epochs", type=int, default=None)
    p.add_argument("--train-batch-size", type=int, default=100)
    p.add_argument("--train-lr", type=float, default=None)
    p.add_argument("--output", default=str(HERE / "results"))
    return p


def sample_sizes(args) -> list[int]:
    if args.sample_sizes is not None:
        values = args.sample_sizes
    elif args.n is not None:
        values = [args.n]
    elif args.protocol == "sow":
        values = {
            "rare_gaussian": [500, 1000, 1500, 2000, 2500],
            "mnist": [200, 400, 600, 800, 1000],
            "cifar10": [250, 500, 750, 1000, 1250],
        }[args.dataset]
    else:
        values = {"blob": [40], "hdgm": [1000], "mnist_fake": [100], "cifar10": [1000]}[args.dataset]
    if any(value < 2 for value in values):
        raise ValueError("Every sample size must be at least two")
    return values


def flatten_pair(x, y, device: torch.device):
    x = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(len(x), -1)
    y = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(len(y), -1)
    return x, y


def effective_ntrees(args, n: int) -> int:
    if args.ntrees is not None:
        return args.ntrees
    if args.protocol == "sow":
        return max(1, math.ceil((n / 2) / args.nlines))
    return 100


def evaluate(method, x, y, args, seed: int, learned, ntrees: int):
    if len(x) != len(y):
        raise ValueError("The permutation test requires equal group sizes")
    z_original = torch.cat((x, y))
    z = z_original.flatten(1)
    generator = torch.Generator(device=z.device).manual_seed(seed + 7919)
    labels = permutation_labels(len(z), len(x), args.permutations, generator, z.device)
    started = time.perf_counter()
    if method in {"mmd", "mmd_rbf"}:
        result = mmd_test(z, labels, args.alpha, kernel="rbf")
    elif method == "mmd_laplace":
        result = mmd_test(z, labels, args.alpha, kernel="laplace")
    elif method == "mmd_o":
        result = mmd_test(z, labels, args.alpha, bandwidth=learned["mmd_o"], kernel="rbf")
    elif method == "mmd_d":
        result = deep_mmd_test(learned["mmd_d"], x, y, labels, args.alpha)
    elif method in {"c2st_s", "c2st_l"}:
        result = c2st_test(learned["c2st"], x, y, labels, args.alpha, hard=method == "c2st_s")
    elif method in {"me", "scf"}:
        result = freqopttest_test(learned[method], x, y, args.alpha)
    elif method == "energy":
        result = energy_test(z, labels, args.alpha)
    elif method == "sw2":
        result = sliced_wasserstein_test(
            z, labels, args.alpha, num_projections=ntrees * args.nlines, seed=seed, p=2.0
        )
    else:
        result = tree_test(
            z, labels, args.alpha, seed=seed, method=method, ntrees=ntrees,
            nlines=args.nlines, delta=args.delta, rho=args.rho,
            fiber_tau=args.fiber_tau, p_agg=args.p_agg,
            num_frequencies=args.num_frequencies, rff_sigma=args.rff_sigma,
            fusion=args.fusion, root_std=args.root_std,
            permutation_chunk=args.permutation_chunk,
        )
    return result, time.perf_counter() - started


def train_learned_methods(args, x_train, y_train):
    learned = {}
    seconds = {method: 0.0 for method in args.methods}
    epochs = args.train_epochs or (2000 if args.dataset == "mnist_fake" else 1000)
    is_image = args.dataset in {"mnist_fake", "cifar10"}
    lr = args.train_lr or (0.0002 if is_image else 0.001)
    if "mmd_o" in args.methods:
        started = time.perf_counter()
        learned["mmd_o"] = train_mmd_o(x_train, y_train, epochs, 0.001 if args.dataset == "hdgm" else 0.0005)
        seconds["mmd_o"] = time.perf_counter() - started
    if "mmd_d" in args.methods:
        started = time.perf_counter()
        learned["mmd_d"] = train_deep_mmd(x_train, y_train, epochs, lr, args.train_batch_size)
        seconds["mmd_d"] = time.perf_counter() - started
    if {"c2st_s", "c2st_l"}.intersection(args.methods):
        started = time.perf_counter()
        learned["c2st"] = train_c2st(x_train, y_train, epochs, lr, args.train_batch_size)
        elapsed = time.perf_counter() - started
        seconds["c2st_s"] = elapsed
        seconds["c2st_l"] = elapsed
    for method in ("me", "scf"):
        if method in args.methods:
            started = time.perf_counter()
            learned[method] = train_freqopttest(x_train, y_train, method, args.alpha)
            seconds[method] = time.perf_counter() - started
    return learned, seconds


def prepare_sow_data(args):
    if args.dataset == "mnist":
        return load_mnist_digit_pools(args.data_dir, args.image_size or 28)
    if args.dataset == "cifar10":
        return load_cifar_pools(args.data_dir, args.cifar101_path, args.image_size or 32, normalize=False)
    return None


def sow_pair(args, pools, n: int, trial: int):
    seed = args.seed + 1_000_003 * n + 10_007 * trial
    if args.dataset == "rare_gaussian":
        return sample_rare_gaussian(n, seed, args.null, args.dimension)
    if args.dataset == "mnist":
        return sample_mnist_mixture_pair(
            *pools, n, seed, null=args.null, contamination=args.contamination
        )
    real, cifar101 = pools
    return sample_image_pair(real, real if args.null else cifar101, n, seed)


def summarize(records, methods):
    summary = {}
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        rate = float(np.mean([record["reject"] for record in selected]))
        se = math.sqrt(rate * (1.0 - rate) / len(selected))
        z = 1.959963984540054
        denominator = 1.0 + z * z / len(selected)
        centre = (rate + z * z / (2.0 * len(selected))) / denominator
        radius = z * math.sqrt(
            rate * (1.0 - rate) / len(selected) + z * z / (4.0 * len(selected) ** 2)
        ) / denominator
        summary[method] = {
            "mean_rejection_rate": rate,
            "binomial_standard_error": se,
            "ci95_wilson": [max(0.0, centre - radius), min(1.0, centre + radius)],
            "mean_runtime_seconds": float(np.mean([record["runtime_seconds"] for record in selected])),
            "mean_training_seconds": float(np.mean([record["training_seconds"] for record in selected])),
        }
    return summary


def save_result(args, n: int, records, summary):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.protocol}_{args.dataset}_{'h0' if args.null else 'h1'}_n{n}"
    csv_path = output / f"{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    json_path = output / f"{tag}_summary.json"
    config = vars(args).copy()
    config["current_n"] = n
    json_path.write_text(json.dumps({"config": config, "results": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {csv_path} and {json_path}", flush=True)


def run_sow(args, sizes, device):
    if args.dataset not in SOW_DATASETS:
        raise ValueError(f"Dataset {args.dataset!r} is not part of the SOW protocol")
    unsupported = LEARNED_METHODS.intersection(args.methods)
    if unsupported:
        raise ValueError(
            "The SOW protocol has no training split; learned methods require --protocol dk: "
            + ", ".join(sorted(unsupported))
        )
    pools = prepare_sow_data(args)
    all_summaries = {}
    for n in sizes:
        ntrees = effective_ntrees(args, n)
        records = []
        print(
            f"n={n} trials={args.trials} permutations={args.permutations} "
            f"ntrees={ntrees} nlines={args.nlines} total_lines={ntrees * args.nlines}",
            flush=True,
        )
        for trial in range(args.trials):
            x, y = flatten_pair(*sow_pair(args, pools, n, trial), device)
            test_seed = args.seed + 1_000_003 * n + 10_007 * trial
            for method in args.methods:
                result, runtime = evaluate(method, x, y, args, test_seed, {}, ntrees)
                record = {
                    "protocol": "sow", "dataset": args.dataset, "null": args.null,
                    "trial": trial, "method": method, "n": n,
                    "dimension": x.shape[1], "statistic": result.statistic,
                    "p_value": result.p_value, "reject": int(result.reject),
                    "runtime_seconds": runtime, "training_seconds": 0.0,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
        summary = summarize(records, args.methods)
        all_summaries[str(n)] = summary
        save_result(args, n, records, summary)
    curve_path = Path(args.output) / f"sow_{args.dataset}_{'h0' if args.null else 'h1'}_power_curve.json"
    curve_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"Saved power curve to {curve_path}", flush=True)


def dk_synthetic_pair(args, n: int, outer: int, test_index: int):
    if args.dataset == "blob":
        seed = 112 * outer + 1 + n if test_index < 0 else 11 * test_index + 10 + n + 10007 * outer
        return sample_blob(n, seed, args.null)
    seed = outer if test_index < 0 else test_index + 2 + 1009 * outer
    return sample_hdgm(n, args.dimension, seed, args.null)


def prepare_dk_image_data(args):
    if args.dataset == "mnist_fake":
        pools = load_mnist_pools(args.data_dir, args.fake_mnist_path, args.image_size or 32)
    else:
        pools = load_cifar_pools(args.data_dir, args.cifar101_path, args.image_size or 64, normalize=True)
    if args.null:
        real, _ = pools
        midpoint = len(real) // 2
        return real[:midpoint], real[midpoint:]
    return pools


def run_dk(args, sizes, device):
    if args.dataset not in DK_DATASETS:
        raise ValueError(f"Dataset {args.dataset!r} is not part of the DK protocol")
    image_pools = prepare_dk_image_data(args) if args.dataset in {"mnist_fake", "cifar10"} else None
    for n in sizes:
        records = []
        ntrees = effective_ntrees(args, n)
        for outer in range(args.outer_trials):
            if image_pools is None:
                x_train, y_train = dk_synthetic_pair(args, n, outer, -1)
                x_train, y_train = flatten_pair(x_train, y_train, device)
            else:
                x_train, y_train, x_pool, y_pool = split_image_pools(*image_pools, n, outer)
                x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
                y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
            learned, training_seconds = train_learned_methods(args, x_train, y_train)
            for test_index in range(args.test_sets):
                if image_pools is None:
                    x, y = dk_synthetic_pair(args, n, outer, test_index)
                    x, y = flatten_pair(x, y, device)
                else:
                    x, y = sample_image_pair(x_pool, y_pool, n, test_index + 1009 * outer)
                    x = torch.as_tensor(x, dtype=torch.float32, device=device)
                    y = torch.as_tensor(y, dtype=torch.float32, device=device)
                test_seed = args.seed + 1_000_003 * outer + 10_007 * test_index
                for method in args.methods:
                    result, runtime = evaluate(method, x, y, args, test_seed, learned, ntrees)
                    records.append({
                        "protocol": "dk", "dataset": args.dataset, "null": args.null,
                        "trial": outer * args.test_sets + test_index, "method": method,
                        "n": len(x), "dimension": x[0].numel(),
                        "statistic": result.statistic, "p_value": result.p_value,
                        "reject": int(result.reject), "runtime_seconds": runtime,
                        "training_seconds": training_seconds[method],
                    })
        save_result(args, n, records, summarize(records, args.methods))


def main():
    args = parser().parse_args()
    if args.methods is None:
        args.methods = ["dbtsw", "rff_fr_rot", "sw2", "mmd_rbf", "mmd_laplace"]
        if args.dataset not in {"mnist", "mnist_fake", "cifar10"}:
            args.methods.insert(1, "fr_rot")
    if args.protocol == "sow" and args.dataset not in SOW_DATASETS:
        raise ValueError(f"Dataset {args.dataset!r} is not part of the SOW protocol")
    if args.protocol == "dk" and args.dataset not in DK_DATASETS:
        raise ValueError(f"Dataset {args.dataset!r} is not part of the DK protocol")
    if args.permutations is None:
        args.permutations = 199 if args.protocol == "sow" else 100
    if args.trials is None:
        args.trials = 200
    if args.dimension is None:
        args.dimension = 15 if args.protocol == "sow" else 10
    if args.p_agg is None:
        args.p_agg = 2.0 if args.protocol == "sow" else 1.0
    if args.permutations < 1 or args.trials < 1 or args.nlines < 1:
        raise ValueError("permutations, trials, and nlines must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    sizes = sample_sizes(args)
    device = torch.device(args.device)
    if args.protocol == "sow":
        run_sow(args, sizes, device)
    else:
        run_dk(args, sizes, device)


if __name__ == "__main__":
    main()
