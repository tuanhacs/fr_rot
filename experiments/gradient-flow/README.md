# Gradient Flow
This is a source code for Gradient Flow task in the paper.

## Requirements
We need python 3.7 or above and these following packages:
```
torch>=1.13.0
matplotlib
scikit-image
scikit-learn
tqdm
numpy
wandb
```

Example command:
```bash
python GradientFlow.py --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 --delta 5.0 --p_tsw 1 --dataset_name "25gaussians" --std 0.001 --rho 0.5 --fiber_tau 1.0 --num_seeds 5
```

`FR_ROT` uses only fiber uncertainty. `rho` must be in `[0, 1]`; setting
`rho=0` recovers the distance-based tree-Wasserstein edge discrepancy. The
residual feature is `r / sqrt(fiber_tau^2 + ||r||^2)`.

```bash
python GradientFlow.py --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 --delta 5.0 --p_tsw 1 --dataset_name "25gaussians" --std 0.001 --noisy_mode interval --lambda_ 0.0001  --i_max 6 --num_seeds 5
```
```bash
python GradientFlow.py --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 --delta 5.0 --p_tsw 1 --dataset_name "25gaussians" --std 0.001 --noisy_mode ball --p_noise 2.0 --lambda_ 0.0001  --i_max 6 --num_seeds 5
```
For Gaussian 30d:
```bash
python GradientFlow.py --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.05 --delta 5.0 --p_tsw 1 --dataset_name "gaussian_30d_small_v" --std 0.001 --noisy_mode interval --lambda_ 0.0001  --i_max 6 --num_seeds 5
```
```bash
python GradientFlow.py --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.05 --delta 5.0 --p_tsw 1 --dataset_name "gaussian_30d_small_v" --std 0.001 --noisy_mode ball --p_noise 2.0 --lambda_ 0.0001  --i_max 6 --num_seeds 5
```
