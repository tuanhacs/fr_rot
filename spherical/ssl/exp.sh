export CUDA_VISIBLE_DEVICES=0
## Ours
python3 train_eval.py --method fr_rot \
    --ntrees 200 --nlines 20 --delta 2 --p 1 \
    --rho 1 --fiber_tau 1 --p_agg 2 \
    --unif_w 2 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0

python3 train_eval.py --method sts_rot \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 2 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0 --noisy_mode interval --lambda_ 0.0001

python3 train_eval.py --method sts_rot \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 2 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0 --noisy_mode ball --lambda_ 0.0001 --p_noise 2.0
