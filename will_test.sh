python3 rl_algos/StableBaselines.py sac fourier_2 --action_space=fourier --fourier_basis_size=2 --num_steps=30000
python3 rl_algos/StableBaselines.py sac fourier_3 --action_space=fourier --fourier_basis_size=3 --num_steps=30000
python3 rl_algos/StableBaselines.py sac fourier_4 --action_space=fourier --fourier_basis_size=4 --num_steps=30000
python3 rl_algos/StableBaselines.py sac fourier_6 --action_space=fourier --fourier_basis_size=6 --num_steps=30000
python3 rl_algos/StableBaselines.py sac fourier_10 --action_space=fourier --fourier_basis_size=12 --num_steps=30000
python3 rl_algos/StableBaselines.py sac fourier_12 --action_space=fourier --fourier_basis_size=12 --num_steps=30000
python3 rl_algos/StableBaselines.py sac standard --num_steps=50000
