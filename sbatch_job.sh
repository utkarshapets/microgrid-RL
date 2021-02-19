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
#SBATCH --ntasks-per-node=24
#
#Request GPUs
#SBATCH --gres=gpu:0
#
#Request CPU
#SBATCH --cpus-per-task=4
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
python StableBaselines.py sac --exp_name=2021_02_18_ms_large_c --reward_function="market_solving" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_medium_c --reward_function="market_solving" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_small_c --reward_function="market_solving" --pb_scenario=3 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_no_batt_c --reward_function="market_solving" --pb_scenario=4 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_no_solar_c --reward_function="market_solving" --pb_scenario=5 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_nothing_c --reward_function="market_solving" --pb_scenario=6 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_large_d --reward_function="market_solving" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_medium_d --reward_function="market_solving" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_small_d --reward_function="market_solving" --pb_scenario=3 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_no_batt_d --reward_function="market_solving" --pb_scenario=4 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_no_solar_d --reward_function="market_solving" --pb_scenario=5 &
python StableBaselines.py sac --exp_name=2021_02_18_ms_nothing_d --reward_function="market_solving" --pb_scenario=6 &


## profit maximizing
python StableBaselines.py sac --exp_name=2021_02_18_pm_large_c --reward_function="profit_maximizing" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_medium_c --reward_function="profit_maximizing" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_small_c --reward_function="profit_maximizing" --pb_scenario=3 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_no_batt_c --reward_function="profit_maximizing" --pb_scenario=4 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_no_solar_c --reward_function="profit_maximizing" --pb_scenario=5 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_nothing_c --reward_function="profit_maximizing" --pb_scenario=6 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_large_d --reward_function="profit_maximizing" --pb_scenario=1 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_medium_d --reward_function="profit_maximizing" --pb_scenario=2 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_small_d --reward_function="profit_maximizing" --pb_scenario=3
python StableBaselines.py sac --exp_name=2021_02_18_pm_no_batt_d --reward_function="profit_maximizing" --pb_scenario=4 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_no_solar_d --reward_function="profit_maximizing" --pb_scenario=5 &
python StableBaselines.py sac --exp_name=2021_02_18_pm_nothing_d --reward_function="profit_maximizing" --pb_scenario=6 &
