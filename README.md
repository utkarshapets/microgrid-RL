# transactive_control
Code meant to support and simulate the Social Game that will be launched in 2020. Elements of transactive control and behavioral engineering will be tested and designed here 

## Installation
1. Clone the repo
2. Install [dvc](https://dvc.org/doc/install) (with google drive support)
  * On linux this is `pip install 'dvc[gdrive]'`
3. Install Docker, if you have not already. 
4. Run `python3 -m dvc remote add -d gdrive gdrive://1qaTn6IYd3cpiyJegDwwEhZ3LwrujK3_x`
5. Run `python3 -m dvc pull`

## Usage
1. Run `./run.sh` from the root of the repo. This will put you in a shell in the docker container with the `rl_algos/logs` directory mounted
2. Run `python3 StableBaselines.py sac test_experiment` in the docker container to start an experiment with the name `test_experiment`
3. Run `tensorboard --logdir rl_algos/logs` from outside the docker container to view the logs

-------

### 9/1/2020

The most recent running of the code involves navigating to the rl_algos/ directory, then running the python command for the vanilla version: 

python StableBaselines.py sac

Adding in the planning model can be done with the following flags:

python StableBaselines.py sac --planning_steps=10 --planning_model=Oracle --num_steps=10000

Please see transactive_control/gym-socialgame/gym_socialgame/envs for files pertaining to the setup of the environment. The socialgame_env.py contains a lot of the information necessary for understanding how the agent steps through the environment. The reward.py file contains information on the variety of reward types available for testing. agents.py contains information on the deterministic people that we created for our simulation. 

###  gym-socialgame
OpenAI Gym environment for a social game.

