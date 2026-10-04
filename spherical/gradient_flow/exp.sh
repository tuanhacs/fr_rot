export CUDA_VISIBLE_DEVICES=1

# Table
python3 main.py -d ssw
python3 main.py -d s3w
python3 main.py -d ri_s3w_1
python3 main.py -d ri_s3w_5
python3 main.py -d ari_s3w
python3 main.py -d stsw --p 1 --delta 50
python3 main.py -d fr_rot --p 1 --rho 1 --fiber_tau 1 --delta 2 --lr 0.05
python3 main.py -d rff_fr_rot --p 1 --rho 1 --fiber_tau 1 --num_frequencies 1 --rff_sigma 1 --delta 2 --lr 0.05
python3 main.py -d sts_rot --noisy_mode interval --lambda_ 0.0001 --delta 1 --lr 0.05
python3 main.py -d sts_rot --noisy_mode ball --p_noise 2.0 --lambda_ 0.0001 --delta 1 --lr 0.05


# Figure
# python3 plot_loss.py
