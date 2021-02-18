#!/bin/bash
# Job name:
#SBATCH --job-name=benchmark_sac_planning
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:0
#
#Request CPU
#SBATCH --cpus-per-task=8
#
# Wall clock limit:
#SBATCH --time=30:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lucas_spangher@berkeley.edu
## Command(s) to run (example):
module load python/3.6
source /global/home/users/lucas_spangher/transactive_control/auto_keras_env/bin/activate
## market_solving
python StableBaselines.py sac --exp_name=2021_02_17_ms_large_a --reward_function="market_solving" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_17_ms_medium_a --reward_function="market_solving" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_17_ms_small_a --reward_function="market_solving" --pb_scenario=3 &
python StableBaselines.py sac --exp_name=2021_02_17_ms_large_b --reward_function="market_solving" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_17_ms_medium_b --reward_function="market_solving" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_17_ms_small_b --reward_function="market_solving" --pb_scenario=3 &
## profit maximizing
python StableBaselines.py sac --exp_name=2021_02_17_pm_large_a --reward_function="market_solving" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_17_pm_medium_a --reward_function="market_solving" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_17_pm_small_a --reward_function="market_solving" --pb_scenario=3 &
python StableBaselines.py sac --exp_name=2021_02_17_pm_large_b --reward_function="market_solving" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_17_pm_medium_b --reward_function="market_solving" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_17_pm_small_b --reward_function="market_solving" --pb_scenario=3
