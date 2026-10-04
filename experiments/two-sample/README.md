# Two-sample testing

The default runner follows Section 4.2 of *Sliced Orlicz-Wasserstein* and
adapts that protocol to DbTSW, linear fiber DbTSW, and random-Fourier fiber
DbTSW.

For every sample size it uses:

- 200 independent trials;
- 199 random label permutations per trial;
- test level `alpha=0.05`;
- equal sample size `n` from each distribution;
- the same observed samples and permutation labels for every method;
- the same tree realization for all tree methods;
- fixed trees and RFF frequencies across the observed and permuted statistics;
- a default tree-line budget `ntrees * nlines ~= n / 2`;
- outer aggregation `p_agg=2`, matching the paper's `SOW_{phi,2}` setup.

The reported rejection rate is test power under H1 and type-I error when
`--null` is supplied.

## SOW protocol

### Rare-mode Gaussian

The paper uses

```text
P = N(0, I_15)
Q = .95 P + .025 N(2.5 e_1, I_15) + .025 N(-2.5 e_1, I_15).
```

Run the Figure-3 sample-size grid:

```bash
python experiments/two-sample/run_tst.py \
  --dataset rare_gaussian \
  --methods dbtsw fr_rot rff_fr_rot sw2 mmd_rbf mmd_laplace \
  --device cuda
```

The default sizes are `500 1000 1500 2000 2500`. A quick single-size run is:

```bash
python experiments/two-sample/run_tst.py \
  --dataset rare_gaussian --n 500 \
  --methods dbtsw rff_fr_rot sw2 mmd_rbf mmd_laplace \
  --device cuda
```

### MNIST digit contamination

The test compares

```text
P = empirical distribution of digit 6
Q = .9 P + .1 empirical distribution of digit 9.
```

MNIST is downloaded by torchvision and pixels are scaled to `[0, 1]`. Run:

```bash
python experiments/two-sample/run_tst.py \
  --dataset mnist \
  --methods dbtsw rff_fr_rot sw2 mmd_rbf mmd_laplace \
  --device cuda
```

The default sizes are `200 400 600 800 1000`. Change the contamination level
with `--contamination`.

### CIFAR-10 versus CIFAR-10.1

Place `cifar10.1_v4_data.npy` under `experiments/two-sample/data`, or pass
its location explicitly:

```bash
python experiments/two-sample/run_tst.py \
  --dataset cifar10 \
  --cifar101-path /path/to/cifar10.1_v4_data.npy \
  --methods dbtsw rff_fr_rot sw2 mmd_rbf mmd_laplace \
  --device cuda
```

The default sizes are `250 500 750 1000 1250`.

Linear `fr_rot` stores a full residual feature for every point and tree line.
On raw images, use `--permutation-chunk 1` or omit it in favor of
`rff_fr_rot` if GPU memory or runtime is limiting. It is therefore excluded
from the default image methods, but can still be requested explicitly.

### Type-I error

Append `--null` to any command. A valid permutation test should have a
rejection rate close to 0.05:

```bash
python experiments/two-sample/run_tst.py \
  --dataset mnist --n 200 --null \
  --methods dbtsw rff_fr_rot sw2 mmd_rbf mmd_laplace \
  --device cuda
```

### Sample sizes are configurable

`n` is not fixed to 100. Use one size:

```bash
--n 500
```

or an arbitrary power curve:

```bash
--sample-sizes 100 200 400 800
```

`--sample-sizes` takes precedence over `--n`.

## Legacy DK-for-TST protocol

The previous Blob, HDGM, real-versus-generated MNIST, learned MMD, and C2ST
pipeline remains available:

```bash
python experiments/two-sample/run_tst.py \
  --protocol dk --dataset blob --n 40 \
  --methods dbtsw fr_rot rff_fr_rot mmd mmd_o mmd_d c2st_l c2st_s \
  --device cuda
```

The former MNIST experiment is now named `mnist_fake`:

```bash
python experiments/two-sample/run_tst.py \
  --protocol dk --dataset mnist_fake --n 100 \
  --fake-mnist-path /path/to/Fake_MNIST_data_EP100_N10000.pckl \
  --methods dbtsw rff_fr_rot mmd energy --device cuda
```

## Outputs

Each sample size produces a CSV containing every test decision and a JSON
summary containing:

- empirical rejection rate;
- binomial standard error and Wilson 95% interval;
- mean test runtime;
- mean training time.

A combined `*_power_curve.json` is also written when using the SOW protocol.
Outputs are stored under `experiments/two-sample/results` unless `--output`
is specified.

## Attribution

The SOW protocol follows *Sliced Orlicz-Wasserstein*. The Blob, HDGM,
real/fake MNIST, learned MMD, and C2ST experiments are modern PyTorch ports of
the MIT-licensed
[DK-for-TST](https://github.com/fengliu90/DK-for-TST) repository accompanying
F. Liu et al., *Learning Deep Kernels for Non-Parametric Two-Sample Tests*,
ICML 2020.
