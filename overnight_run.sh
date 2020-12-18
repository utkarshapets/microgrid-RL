python3 rl_algos/StableBaselines.py sac standard --one_day=15 --num_steps=200000

python3 rl_algos/StableBaselines.py sac fourier_02 --action_space=fourier --fourier_basis_size=02 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_03 --action_space=fourier --fourier_basis_size=03 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_04 --action_space=fourier --fourier_basis_size=04 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_05 --action_space=fourier --fourier_basis_size=05 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_07 --action_space=fourier --fourier_basis_size=07 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_10 --action_space=fourier --fourier_basis_size=10 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_12 --action_space=fourier --fourier_basis_size=12 --one_day=15 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac fourier_20 --action_space=fourier --fourier_basis_size=20 --one_day=15 --pricing_type=TOU --num_steps=200000

python3 rl_algos/StableBaselines.py sac normalized --action_space=c_norm --one_day=15 --pricing_type=TOU --num_steps=200000

python3 rl_algos/StableBaselines.py sac tou_mag_1_2       --action_space=c      --one_day=15 --manual_tou_magnitude=01.2 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_2_0       --action_space=c      --one_day=15 --manual_tou_magnitude=02.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_2_5       --action_space=c      --one_day=15 --manual_tou_magnitude=02.5 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_3_0       --action_space=c      --one_day=15 --manual_tou_magnitude=03.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_4_5       --action_space=c      --one_day=15 --manual_tou_magnitude=04.5 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_6_0       --action_space=c      --one_day=15 --manual_tou_magnitude=06.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_8_0       --action_space=c      --one_day=15 --manual_tou_magnitude=08.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_10_0      --action_space=c      --one_day=15 --manual_tou_magnitude=10.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_1_2_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=01.2 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_2_0_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=02.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_2_5_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=02.5 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_3_0_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=03.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_4_5_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=04.5 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_6_0_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=06.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_8_0_norm  --action_space=c_norm --one_day=15 --manual_tou_magnitude=08.0 --pricing_type=TOU --num_steps=200000
python3 rl_algos/StableBaselines.py sac tou_mag_10_0_norm --action_space=c_norm --one_day=15 --manual_tou_magnitude=10.0 --pricing_type=TOU --num_steps=200000

python3 rl_algos/StableBaselines.py sac standard --action_space=c --one_day=15 --pricing_type=TOU --num_steps=1000000
