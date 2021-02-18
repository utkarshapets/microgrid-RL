import argparse
import numpy as np
import gym
from stableBaselines.stable_baselines.common.vec_env import (  # pylint: disable=import-error, no-name-in-module
    DummyVecEnv,
    VecCheckNan,
    VecNormalize,
)
from stableBaselines.stable_baselines.common.evaluation import (  # pylint: disable=import-error, no-name-in-module
    evaluate_policy,
)
from stableBaselines.stable_baselines.common.env_checker import (  # pylint: disable=import-error, no-name-in-module
    check_env,
)

import sys
sys.path.insert(0, '..')

# import gym_microgrid.gym_microgrid.envs.utils as env_utils

import utils as env_utils

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorboard_logger import (  # pylint: disable=import-error, no-name-in-module
    configure as tb_configure,
)
from tensorboard_logger import (  # pylint: disable=import-error, no-name-in-module
    log_value as tb_log_value,
)

import utils

import os


def train(agent, num_steps, planning_steps, tb_log_name):
    """
    Purpose: Train agent in env, and then call eval function to evaluate policy
    """
    # Train agent

    agent.learn(
        total_timesteps=num_steps,
        log_interval=10,
        planning_steps=planning_steps,
        tb_log_name=tb_log_name
    )


def eval_policy(model, env, num_eval_episodes: int, list_reward_per_episode=False):
    """
    Purpose: Evaluate policy on environment over num_eval_episodes and print results

    Args:
        Model: Stable baselines model
        Env: Gym environment for evaluation
        num_eval_episodes: (Int) number of episodes to evaluate policy
        list_reward_per_episode: (Boolean) Whether or not to return a list containing rewards per episode (instead of mean reward over all episodes)

    """
    mean_reward, std_reward = evaluate_policy(
        model, env, num_eval_episodes, return_episode_rewards=list_reward_per_episode
    )

    print("Test Results: ")
    print("Mean Reward: {:.3f}".format(mean_reward))
    print("Std Reward: {:.3f}".format(std_reward))


def get_agent(env, args, non_vec_env=None):
    """
    Purpose: Import algo, policy and create agent

    Returns: Agent

    Exceptions: Raises exception if args.algo unknown (not needed b/c we filter in the parser, but I added it for modularity)
    """
    if args.algo == "sac":
        from stableBaselines.stable_baselines.sac.sac import SAC as mySAC
        from stable_baselines.sac.policies import MlpPolicy as policy
        plotter_person_reaction = utils.plotter_person_reaction
        action_to_prices_fn = lambda x: (x + 1) * 5 #normal continuous

        return mySAC(
            policy,
            env,
            non_vec_env=non_vec_env,
            batch_size=args.batch_size,
            learning_starts=30,
            verbose=0,
            tensorboard_log=args.rl_log_path,
            people_reaction_log_dir=os.path.join(args.log_path, "people_reaction/"),
            plotter_person_reaction=plotter_person_reaction,
            action_to_prices_fn=action_to_prices_fn,
            learning_rate=args.learning_rate
        )

    else:
        raise NotImplementedError("Algorithm {} not supported. :( ".format(args.algo))


def args_convert_bool(args):
    """
    Purpose: Convert args which are specified as strings (e.g. yesterday, energy) into boolean to work with environment
    """
    if not isinstance(args.yesterday, (bool)):
        args.yesterday = utils.string2bool(args.yesterday)
    if not isinstance(args.energy, (bool)):
        args.energy = utils.string2bool(args.energy)
    if not isinstance(args.test_planning_env, (bool)):
        args.test_planning_env = utils.string2bool(args.test_planning_env)

def get_environment(args, include_non_vec_env=False):
    """
    Purpose: Create environment for algorithm given by args. algo

    Args:
        args

    Returns: Environment with action space compatible with algo
    """
    # Convert string args (which are supposed to be bool) into actual boolean values

    print(args.planning_steps, args.test_planning_env)
    planning = (args.planning_steps > 0) or args.test_planning_env

    # SAC only works in continuous environment
    if args.algo == "sac":        
        action_space_string = "continuous"
    # For algos (e.g. ppo) which can handle discrete or continuous case
    # Note: PPO typically uses normalized environment (#TODO)
    else:
        convert_action_space_str = (
            lambda s: "continuous" if s == "c" else "multidiscrete"
        )
        action_space_string = convert_action_space_str(args.action_space)

    planning_flag = args.planning_steps > 0

    if args.env_id == "hourly":
        env_id = "_hourly-v0"
    elif args.env_id == "monthly":
        env_id = "_monthly-v0"
    else:
        env_id = "-v0"

    if not planning:
        microgrid_env = gym.make(
            "gym_microgrid:microgrid{}".format(env_id),
            action_space_string=action_space_string,
            response_type_string=args.response,
            one_day=args.one_day,
            number_of_participants=args.num_players,
            yesterday_in_state=args.yesterday,
            energy_in_state=args.energy,
            pricing_type=args.pricing_type,
            reward_function=args.reward_function,
            fourier_basis_size=args.fourier_basis_size,
            manual_tou_magnitude=args.manual_tou_magnitude,
            complex_batt_pv_scenario = args.pb_scenario,
        )
    else:
        # go into the planning mode
        microgrid_env = gym.make(
            "gym_microgrid:microgrid{}".format("_planning-v0"),
            action_space_string=action_space_string,
            response_type_string=args.response,
            one_day=args.one_day,
            number_of_participants=args.num_players,
            yesterday_in_state=args.yesterday,
            energy_in_state=args.energy,
            pricing_type=args.pricing_type,
            planning_flag=planning_flag,
            planning_steps=args.planning_steps,
            planning_model_type=args.planning_model,
            own_tb_log=args.rl_log_path,
            reward_function=reward_function
        )

    # Check to make sure any new changes to environment follow OpenAI Gym API
    check_env(microgrid_env)

    # temp_step_fnc = microgrid_env.step

    # Using env_fn so we can create vectorized environment.
    env_fn = lambda: microgrid_env
    venv = DummyVecEnv([env_fn])
    env = VecNormalize(venv)

    # env.step = temp_step_fnc
    if not include_non_vec_env:
        return env
    else:
        return env, microgrid_env


def parse_args():
    """
    Purpose: Parse arguments to run script
    """

    parser = argparse.ArgumentParser(
        description="Arguments for running Stable Baseline RL Algorithms on microgridEnv"
    )
    parser.add_argument(
        "--env_id",
        help="Environment ID for Gym Environment",
        type=str,
        choices=["v0", "monthly"],
        default="v0",
    )
    parser.add_argument(
        "algo", help="Stable Baselines Algorithm", type=str, choices=["sac", "ppo"]
    )
    parser.add_argument(
        "--exp_name", help="Name of the experiment. Used to name log files, etc.", type=str
    )
    parser.add_argument(
        "--base_log_dir",
        help="Base directory for tensorboard logs",
        type=str,
        default="./logs/"
    )

    parser.add_argument(
        "--batch_size",
        help="Batch Size for sampling from replay buffer",
        type=int,
        default=5,
        choices=[i for i in range(1, 30)],
    )
    parser.add_argument(
        "--num_steps",
        help="Number of timesteps to train algo",
        type=int,
        default=100000,
    )
    # Note: only some algos (e.g. PPO) can use LSTM Policy the feature below is for future testing
    parser.add_argument(
        "--policy_type",
        help="Type of Policy (e.g. MLP, LSTM) for algo",
        default="mlp",
        choices=["mlp", "lstm"],
    )
    parser.add_argument(
        "--action_space",
        help="Action Space for Algo (only used for algos that are compatable with both discrete & cont",
        default="c",
        choices=["c", "c_norm", "d", "fourier"],
    )
    parser.add_argument(
        "--fourier_basis_size",
        help="Fourier basis size to use when using fourier action space",
        type=int,
        default=4,
        choices=list(range(100))
    )
    parser.add_argument(
        "--response",
        help="Player response function (l = linear, t = threshold_exponential, s = sinusoidal",
        type=str,
        default="l",
        choices=["l", "t", "s"],
    )
    parser.add_argument(
        "--one_day",
        help="Specific Day of the year to Train on (default = None, train over entire yr)",
        type=int,
        default=0,
        choices=[i for i in range(-1, 366)],
    )
    parser.add_argument(
        "--manual_tou_magnitude",
        help="Relative magnitude of the TOU (should be > 1)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--num_players",
        help="Number of players ([1, 20]) in social game",
        type=int,
        default=10,
        choices=[i for i in range(1, 21)],
    )
    parser.add_argument(
        "--yesterday",
        help="Whether to include yesterday in state (default = F)",
        type=str,
        default="F",
        choices=["T", "F"],
    )
    parser.add_argument(
        "--energy",
        help="Whether to include energy in state (default = F)",
        type=str,
        default="F",
        choices=["T", "F"],
    )
    parser.add_argument(
        "--planning_steps",
        help="How many planning iterations to partake in",
        type=int,
        default=0,
        choices=[i for i in range(0, 100)],
    )
    parser.add_argument(
        "--planning_model",
        help="Which planning model to use",
        type=str,
        default="Oracle",
        choices=["Oracle", "Baseline", "LSTM", "OLS"],
    )
    parser.add_argument(
        "--pricing_type",
        help="time of use or real time pricing",
        type=str,
        choices=["TOU", "RTP"],
        default="TOU",
    )
    parser.add_argument(
        "--test_planning_env",
        help="flag if you want to test vanilla planning",
        type=str,
        default='F',
        choices=['T', 'F'],
    )
    parser.add_argument(
        "--reward_function",
        help="reward function to test",
        type=str,
        default="market_solving",
        choices = ["market_solving", "profit_maximizing"],
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate of the the agent",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--pb_scenario",
        type=int,
        default = 1,
        help="1 is for repeated PV, 2 for small, 3 or medium scenario",
        choices=[1,2,3])
    args = parser.parse_args()

    args.log_path = os.path.join(args.base_log_dir, args.exp_name + "/")
    args.rl_log_path = os.path.join(args.log_path, "rl/")


    return args


def main():
    # Get args
    args = parse_args()

    # Print args for reference
    print(args)
    args_convert_bool(args)

    # Create environments

    if os.path.exists(args.log_path):
        print("Choose a new name for the experiment, log dir already exists")
        raise ValueError

    env, microgrid_env = get_environment(
        args, include_non_vec_env=True
    )
    print("Got environment, getting agent")

    # Create Agent
    model = get_agent(env, args, non_vec_env=microgrid_env)
    print("Got agent")

    # Train algo, (logging through Tensorboard)
    print("Beginning Testing!")
    r_real = train(
        model,
        args.num_steps,
        planning_steps=args.planning_steps,
        tb_log_name=args.exp_name
    )

    print("Training Completed! View TensorBoard logs at " + args.log_path)

    # Print evaluation of policy
    print("Beginning Evaluation")

    eval_env = get_environment(args)
    eval_policy(model, eval_env, num_eval_episodes=10)

    print(
        "If there was no planning model involved, remember that the output will be in the log dir"
    )


if __name__ == "__main__":
    main()
