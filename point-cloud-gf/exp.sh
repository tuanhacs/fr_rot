
python main_point.py --losses ts_rot --noisy_mode interval --lambda_ 0.0001  --data ./reconstruct_random_50_shapenetcore55.npy
python main_point.py --losses ts_rot --noisy_mode ball --p_noise 2.0 --lambda_ 0.0001  --data ./reconstruct_random_50_shapenetcore55.npy
python main_point.py --losses twd --p 1  --data ./reconstruct_random_50_shapenetcore55.npy
python main_point.py --losses fr_rot --rho 1.0 --fiber_tau 1.0 --p_agg 2.0 --data ./reconstruct_random_50_shapenetcore55.npy
