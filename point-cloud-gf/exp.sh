
python main_point.py --losses ts_rot --noisy_mode interval --lambda_ 0.0001  --data ./reconstruct_random_50_shapenetcore55.npy
python main_point.py --losses ts_rot --noisy_mode ball --p_noise 2.0 --lambda_ 0.0001  --data ./reconstruct_random_50_shapenetcore55.npy
python main_point.py --losses twd --p 1  --data ./reconstruct_random_50_shapenetcore55.npy