python3 rl_algos/StableBaselines.py sac fourier_4_alpha_3  --action_space=fourier --fourier_basis_size=4  --num_steps=100000 --learning_rate=3e-3
python3 rl_algos/StableBaselines.py sac standard_alpha_3 --num_steps=100000 --learning_rate=3e-3
python3 rl_algos/StableBaselines.py sac fourier_4_alpha_4  --action_space=fourier --fourier_basis_size=4  --num_steps=100000 --learning_rate=3e-4
python3 rl_algos/StableBaselines.py sac standard_alpha_4 --num_steps=100000 --learning_rate=3e-4
python3 rl_algos/StableBaselines.py sac fourier_4_alpha_5  --action_space=fourier --fourier_basis_size=4  --num_steps=100000 --learning_rate=3e-5
python3 rl_algos/StableBaselines.py sac standard_alpha_5 --num_steps=100000 --learning_rate=3e-5
python3 rl_algos/StableBaselines.py sac fourier_4_alpha_6  --action_space=fourier --fourier_basis_size=4  --num_steps=100000 --learning_rate=3e-6
python3 rl_algos/StableBaselines.py sac standard_alpha_6 --num_steps=100000 --learning_rate=3e-6
