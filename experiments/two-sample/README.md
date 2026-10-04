# Two-sample testing

This experiment ports the data construction and power-evaluation protocol from
[DK-for-TST](https://github.com/fengliu90/DK-for-TST) to current PyTorch. It
adds DbTSW, linear fiber DbTSW, and random-Fourier fiber DbTSW to a shared,
label-permutation test.

It also provides current-PyTorch ports of the main learned baselines from the
reference pipeline: optimized-bandwidth MMD (`mmd_o`), deep-kernel MMD
(`mmd_d`), and soft/hard classifier tests (`c2st_l`, `c2st_s`). Plain
median-bandwidth MMD and energy distance are available as training-free checks.
The original ME and SCF baselines are exposed as optional methods through
`freqopttest`.

The original defaults are retained:

- test level `alpha=0.05`;
- 100 label permutations;
- 10 outer trials;
- 100 independently sampled test sets per outer trial;
- Blob: 40 observations per mode, hence 360 observations per group;
- HDGM: 1000 observations per mode and two modes;
- MNIST: 100 observations per group;
- CIFAR-10/CIFAR-10.1: 1000 observations per group.

Unlike the legacy code, the implementation uses the finite-sample permutation
p-value `(1 + exceedances) / (1 + permutations)`, supports CPU or CUDA, and
caches every label-independent tree calculation. Trees and RFF frequencies are
fixed within a permutation test.

## Quick smoke runs

```bash
python experiments/two-sample/run_tst.py --dataset blob \
  --methods dbtsw fr_rot rff_fr_rot mmd energy \
  --n 5 --ntrees 4 --nlines 2 --permutations 19 \
  --outer-trials 1 --test-sets 2 --device cpu

python experiments/two-sample/run_tst.py --dataset hdgm \
  --methods dbtsw fr_rot rff_fr_rot mmd energy \
  --n 1000 --dimension 10 --device cuda
```

Add `--null` to estimate type-I error instead of power.

Learned baselines are opt-in because their paper defaults train for 1000 or
2000 epochs in every outer trial:

```bash
python experiments/two-sample/run_tst.py --dataset blob \
  --methods mmd_o mmd_d c2st_l c2st_s --device cuda
```

Use `--train-epochs` for a smoke run; omitting it restores the paper defaults.

For ME and SCF, install the same dependency used by the reference repository:

```bash
pip install git+https://github.com/wittawatj/interpretable-test
```

## MNIST

Download `Fake_MNIST_data_EP100_N10000.pckl` from the link in the
[DK-for-TST README](https://github.com/fengliu90/DK-for-TST#download-data), then
run:

```bash
python experiments/two-sample/run_tst.py --dataset mnist \
  --fake-mnist-path /path/to/Fake_MNIST_data_EP100_N10000.pckl \
  --methods dbtsw fr_rot rff_fr_rot mmd energy --device cuda
```

The loader uses the first 4000 real training images and first 4000 generated
images, then creates disjoint train/test pools for every outer trial, matching
the reference protocol.

## CIFAR-10 versus CIFAR-10.1

Use `cifar10.1_v4_data.npy` from
[DK-for-TST](https://github.com/fengliu90/DK-for-TST/blob/master/cifar10.1_v4_data.npy):

```bash
python experiments/two-sample/run_tst.py --dataset cifar10 \
  --cifar101-path /path/to/cifar10.1_v4_data.npy \
  --methods dbtsw rff_fr_rot mmd energy --device cuda
```

The legacy code evaluates all remaining CIFAR-10.1 images at once after using
1000 images for training. That makes group size depend on the pool and contains
an incorrect C2ST group-size argument. This port uses the documented `n`
observations per group for every independent test set.

For raw images, linear `fr_rot` materializes a residual vector for every
tree/branch/point and is intentionally not in the recommended command.
`rff_fr_rot` computes residual Fourier phases without storing that tensor.

## Important options

- `--fusion max` reproduces the current fiber implementation;
- `--fusion add` evaluates the additive fiber IPM;
- `--rho`, `--fiber-tau`, `--num-frequencies`, `--rff-sigma` configure fibers;
- `--ntrees`, `--nlines`, `--delta` configure the common tree system;
- `--root-std 0` places roots at the pooled mean, using labels neither for tree
  construction nor permutation calibration.

Each run writes all individual decisions to CSV and aggregate power/type-I
error to JSON under `experiments/two-sample/results`.

## Attribution

The Blob and HDGM distributions, image preprocessing, default sample sizes,
outer repetitions, and seed structure are adapted from the MIT-licensed
DK-for-TST repository accompanying:

F. Liu et al., *Learning Deep Kernels for Non-Parametric Two-Sample Tests*,
ICML 2020.
