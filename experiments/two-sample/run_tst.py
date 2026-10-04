"""Run DK-for-TST-style power experiments with modern PyTorch.

Defaults preserve the original protocol: alpha=.05, 100 permutations, ten
outer trials, and 100 independently sampled test sets per outer trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from datasets import (  # noqa: E402
    load_cifar_pools,
    load_mnist_pools,
    sample_blob,
    sample_hdgm,
    sample_image_pair,
    split_image_pools,
)
from tst_statistics import (  # noqa: E402
    energy_test,
    mmd_test,
    permutation_labels,
    tree_test,
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


TREE_METHODS = {"dbtsw", "fr_rot", "rff_fr_rot"}
ALL_METHODS = ("dbtsw", "fr_rot", "rff_fr_rot", "mmd", "mmd_o", "mmd_d", "energy", "c2st_s", "c2st_l", "me", "scf")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("blob", "hdgm", "mnist", "cifar10"), required=True)
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=["dbtsw", "fr_rot", "rff_fr_rot", "mmd", "energy"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--outer-trials", type=int, default=10)
    p.add_argument("--test-sets", type=int, default=100)
    p.add_argument("--seed", type=int, default=1102)
    p.add_argument("--null", action="store_true", help="Estimate type-I error instead of power")
    p.add_argument("--n", type=int, default=None, help="Samples per mode for synthetic data or per image group")
    p.add_argument("--dimension", type=int, default=10, help="HDGM dimension")
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--fake-mnist-path", default=str(HERE / "data" / "Fake_MNIST_data_EP100_N10000.pckl"))
    p.add_argument("--cifar101-path", default=str(HERE / "data" / "cifar10.1_v4_data.npy"))
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--ntrees", type=int, default=100)
    p.add_argument("--nlines", type=int, default=4)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--fiber-tau", type=float, default=1.0)
    p.add_argument("--p-agg", type=float, default=1.0)
    p.add_argument("--num-frequencies", type=int, default=4)
    p.add_argument("--rff-sigma", type=float, default=1.0)
    p.add_argument("--fusion", choices=("max", "add"), default="max")
    p.add_argument("--root-std", type=float, default=0.0)
    p.add_argument("--permutation-chunk", type=int, default=16)
    p.add_argument("--train-epochs", type=int, default=None, help="MMD-O/MMD-D/C2ST epochs; dataset paper defaults when omitted")
    p.add_argument("--train-batch-size", type=int, default=100)
    p.add_argument("--train-lr", type=float, default=None)
    p.add_argument("--output", default=str(HERE / "results"))
    return p


def default_n(dataset: str) -> int:
    # Original defaults: Blob 40 samples/mode, HDGM 1000 samples/mode,
    # MNIST 100/group, CIFAR-10 1000/group.
    return {"blob": 40, "hdgm": 1000, "mnist": 100, "cifar10": 1000}[dataset]


def flatten_pair(x, y, device: str):
    x = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(len(x), -1)
    y = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(len(y), -1)
    return x, y


def synthetic_pair(args, outer: int, test_index: int):
    # These formulas retain the original deterministic separation between the
    # outer training trial and its independent power-evaluation samples.
    if args.dataset == "blob":
        seed = 112 * outer + 1 + args.n if test_index < 0 else 11 * test_index + 10 + args.n + 10007 * outer
        return sample_blob(args.n, seed, args.null)
    seed = outer if test_index < 0 else (test_index + 2 + 1009 * outer)
    return sample_hdgm(args.n, args.dimension, seed, args.null)


def load_image_data(args):
    size = args.image_size or (32 if args.dataset == "mnist" else 64)
    if args.dataset == "mnist":
        real, shifted = load_mnist_pools(args.data_dir, args.fake_mnist_path, size)
    else:
        real, shifted = load_cifar_pools(args.data_dir, args.cifar101_path, size)
    if args.null:
        midpoint = len(real) // 2
        return real[:midpoint], real[midpoint:]
    return real, shifted


def evaluate(method: str, x: torch.Tensor, y: torch.Tensor, args, seed: int, learned):
    if len(x) != len(y):
        raise ValueError("This reproduction pipeline expects equal group sizes")
    z_original = torch.cat((x, y))
    z = z_original.flatten(1)
    label_generator = torch.Generator(device=z.device).manual_seed(seed + 7919)
    labels = permutation_labels(len(z), len(x), args.permutations, label_generator, z.device)
    started = time.perf_counter()
    if method == "mmd":
        result = mmd_test(z, labels, args.alpha)
    elif method == "mmd_o":
        result = mmd_test(z, labels, args.alpha, bandwidth=learned["mmd_o"])
    elif method == "mmd_d":
        result = deep_mmd_test(learned["mmd_d"], x, y, labels, args.alpha)
    elif method in {"c2st_s", "c2st_l"}:
        result = c2st_test(learned["c2st"], x, y, labels, args.alpha, hard=method == "c2st_s")
    elif method in {"me", "scf"}:
        result = freqopttest_test(learned[method], x, y, args.alpha)
    elif method == "energy":
        result = energy_test(z, labels, args.alpha)
    else:
        result = tree_test(
            z,
            labels,
            args.alpha,
            seed=seed,
            method=method,
            ntrees=args.ntrees,
            nlines=args.nlines,
            delta=args.delta,
            rho=args.rho,
            fiber_tau=args.fiber_tau,
            p_agg=args.p_agg,
            num_frequencies=args.num_frequencies,
            rff_sigma=args.rff_sigma,
            fusion=args.fusion,
            root_std=args.root_std,
            permutation_chunk=args.permutation_chunk,
        )
    return result, time.perf_counter() - started


def train_learned_methods(args, x_train, y_train):
    learned = {}
    training_seconds = {method: 0.0 for method in args.methods}
    epochs = args.train_epochs or (2000 if args.dataset == "mnist" else 1000)
    lr = args.train_lr or (0.0002 if args.dataset in {"mnist", "cifar10"} else 0.001)
    if "mmd_o" in args.methods:
        started = time.perf_counter()
        learned["mmd_o"] = train_mmd_o(x_train, y_train, epochs, 0.0005 if args.dataset != "hdgm" else 0.001)
        training_seconds["mmd_o"] = time.perf_counter() - started
    if "mmd_d" in args.methods:
        started = time.perf_counter()
        learned["mmd_d"] = train_deep_mmd(x_train, y_train, epochs, lr, args.train_batch_size)
        training_seconds["mmd_d"] = time.perf_counter() - started
    if {"c2st_s", "c2st_l"}.intersection(args.methods):
        started = time.perf_counter()
        learned["c2st"] = train_c2st(x_train, y_train, epochs, lr, args.train_batch_size)
        elapsed = time.perf_counter() - started
        training_seconds["c2st_s"] = elapsed
        training_seconds["c2st_l"] = elapsed
    for method in ("me", "scf"):
        if method in args.methods:
            started = time.perf_counter()
            learned[method] = train_freqopttest(x_train, y_train, method, args.alpha)
            training_seconds[method] = time.perf_counter() - started
    return learned, training_seconds


def main():
    args = parser().parse_args()
    args.n = args.n or default_n(args.dataset)
    if args.permutations < 1 or args.outer_trials < 1 or args.test_sets < 1:
        raise ValueError("permutations, outer-trials and test-sets must be positive")
    device = torch.device(args.device)
    image_data = load_image_data(args) if args.dataset in {"mnist", "cifar10"} else None
    records = []

    for outer in range(args.outer_trials):
        if image_data is not None:
            real_train, shifted_train, real_test, shifted_test = split_image_pools(*image_data, args.n, outer)
        else:
            real_train, shifted_train = synthetic_pair(args, outer, -1)
        real_train, shifted_train = flatten_pair(real_train, shifted_train, device) if image_data is None else (
            torch.as_tensor(real_train, dtype=torch.float32, device=device),
            torch.as_tensor(shifted_train, dtype=torch.float32, device=device),
        )
        learned, training_seconds = train_learned_methods(args, real_train, shifted_train)
        for test_index in range(args.test_sets):
            if image_data is None:
                x, y = synthetic_pair(args, outer, test_index)
            else:
                test_n = args.n
                x, y = sample_image_pair(real_test, shifted_test, test_n, test_index + 1009 * outer)
            if image_data is None:
                x, y = flatten_pair(x, y, device)
            else:
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
                y = torch.as_tensor(y, dtype=torch.float32, device=device)
            for method_index, method in enumerate(args.methods):
                # All methods receive exactly the same label permutations. All
                # tree methods additionally receive the same tree system.
                method_seed = args.seed + 1_000_003 * outer + 10_007 * test_index
                result, runtime = evaluate(method, x, y, args, method_seed, learned)
                record = {
                    "dataset": args.dataset,
                    "null": args.null,
                    "outer_trial": outer,
                    "test_set": test_index,
                    "method": method,
                    "n": len(x),
                    "dimension": x[0].numel(),
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "reject": int(result.reject),
                    "runtime_seconds": runtime,
                    "training_seconds": training_seconds[method],
                }
                records.append(record)
                print(json.dumps(record), flush=True)

        for method in args.methods:
            current = [r["reject"] for r in records if r["outer_trial"] == outer and r["method"] == method]
            print(f"outer={outer} method={method} rejection_rate={np.mean(current):.4f}", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{'h0' if args.null else 'h1'}_n{args.n}"
    csv_path = output_dir / f"{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    summary = {}
    for method in args.methods:
        selected = [r for r in records if r["method"] == method]
        rates = []
        for outer in range(args.outer_trials):
            rates.append(np.mean([r["reject"] for r in selected if r["outer_trial"] == outer]))
        summary[method] = {
            "mean_rejection_rate": float(np.mean(rates)),
            "std_across_outer_trials": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in selected])),
            "mean_training_seconds_per_outer": float(np.mean([
                next(r["training_seconds"] for r in selected if r["outer_trial"] == outer)
                for outer in range(args.outer_trials)
            ])),
        }
    summary_path = output_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps({"config": vars(args), "results": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
