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
#SBATCH --cpus-per-task=1
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
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_large_a --reward_function="market_solving" --pb_scenario=1 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_medium_a --reward_function="market_solving" --pb_scenario=2 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_small_a --reward_function="market_solving" --pb_scenario=3 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_no_batt_a --reward_function="market_solving" --pb_scenario=4 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_no_solar_a --reward_function="market_solving" --pb_scenario=5 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_nothing_a --reward_function="market_solving" --pb_scenario=6 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_large_b --reward_function="market_solving" --pb_scenario=1 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_medium_b --reward_function="market_solving" --pb_scenario=2 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_small_b --reward_function="market_solving" --pb_scenario=3 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_no_batt_b --reward_function="market_solving" --pb_scenario=4 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_no_solar_b --reward_function="market_solving" --pb_scenario=5 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_ms_nothing_b --reward_function="market_solving" --pb_scenario=6 --two_price_state=T &


## profit maximizing
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_large_a --reward_function="profit_maximizing" --pb_scenario=1 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_medium_a --reward_function="profit_maximizing" --pb_scenario=2 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_small_a --reward_function="profit_maximizing" --pb_scenario=3 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_no_batt_a --reward_function="profit_maximizing" --pb_scenario=4 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_no_solar_a --reward_function="profit_maximizing" --pb_scenario=5 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_nothing_a --reward_function="profit_maximizing" --pb_scenario=6 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_large_b --reward_function="profit_maximizing" --pb_scenario=1 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_medium_b --reward_function="profit_maximizing" --pb_scenario=2 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_small_b --reward_function="profit_maximizing" --pb_scenario=3 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_no_batt_b --reward_function="profit_maximizing" --pb_scenario=4 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_no_solar_b --reward_function="profit_maximizing" --pb_scenario=5 --two_price_state=T &
python StableBaselines.py sac --exp_name=2021_02_19_twoprice_pm_nothing_b --reward_function="profit_maximizing" --pb_scenario=6 --two_price_state=T 
