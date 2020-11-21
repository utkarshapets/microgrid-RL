import gym
from gym import spaces

import numpy as np

from gym_socialgame.envs.socialgame_env import SocialGameEnv
from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward

import tensorflow as tf

import pickle


class SocialGamePlanningEnv(SocialGameEnv):
    def __init__(self,
        action_space_string = "continuous",
        response_type_string = "l",
        number_of_participants = 10,
        one_day = 0,
        energy_in_state = False,
        yesterday_in_state = False,
        day_of_week = False,
        pricing_type= "TOU",
        reward_function = "scaled_cost_distance",
        planning_flag = False,
        planning_steps = 0,
        planning_model_type = "Oracle",
        own_tb_log = None):

        super().__init__(action_space_string,
        response_type_string,
        number_of_participants,
        one_day,
        energy_in_state,
        yesterday_in_state,
        day_of_week,
        pricing_type,
        reward_function
        )

        self.planning_flag = planning_flag
        self.planning_steps = planning_steps
        self.planning_model_type = planning_model_type

    def _planning_prediction(
        self, action, day_of_week, planning_model_type="OLS", loaded_model=None,
    ):

        """
        Function for calling the planning model and producing an average response

        Inputs:

        Action: [10-float] a list of 10 floats that are the points provided by the agent to the env
        day_of_week: [int] a number indicating the day of the week
        planning_model_type: str, either "Oracle" for a perfect planning model,
            "LSTM" for the rnn implementation, "OLS" for linear regression, or "
            baseline" for a mean estimate
        loaded_model: pass in a loaded model

        """

        # if self.min_demand is not None and self.max_demand is not None:
        #     scaler = MinMaxScaler(feature_range = (self.min_demand, self.max_demand))

        energy_consumptions = {}
        total_consumption = np.zeros(10)

        if planning_model_type == "Oracle":
            prev_observation = self.prices[(self.day)]
            energy_consumptions = self._simulate_humans(action)
            return energy_consumptions

        ## Basic LSTM that follows the rules of the experiment
        elif planning_model_type == "LSTM":
            ## load the minMaxScalers
            with open("scaler_X.pickle", "rb") as input_file:
                scaler_X = pickle.load(input_file)
            with open("scaler_y.pickle", "rb") as input_file:
                scaler_y = pickle.load(input_file)

            ## prepare the data

            d_X = pd.DataFrame(data={"action": action, "dow": day_of_week})
            scaled_X = scaler_X.transform(d_X)
            sxr = scaled_X.reshape((scaled_X.shape[0], 1, scaled_X.shape[1]))
            print(sxr)

            for player_name in self.player_dict:

                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()

                preds = loaded_model.predict(sxr)

                inv_preds = scaler_y.inverse_transform(preds)

                scaler = MinMaxScaler((player_min_demand, player_max_demand))
                inv_preds = scaler.fit_transform(inv_preds.reshape(-1, 1))

                energy_consumptions[player_name] = np.squeeze(inv_preds)

                total_consumption += np.squeeze(inv_preds)

            energy_consumptions["avg"] = total_consumption / self.number_of_participants
            # print(energy_consumptions["avg"])
            return energy_consumptions

        # simple OLS trained on small dataset without IV
        elif planning_model_type == "OLS":

            for player_name in self.player_dict:

                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()

                energy = 246 + -3.26 * np.array(action)

                scaler = MinMaxScaler((player_min_demand, player_max_demand))
                energy = scaler.fit_transform(energy.reshape(-1, 1))
                energy = np.squeeze(energy)

                energy_consumptions[player_name] = energy
                total_consumption += energy

            energy_consumptions["avg"] = total_consumption / self.number_of_participants

            return energy_consumptions

        # baseline model that just returns average of the sample energy day
        elif planning_model_type == "Baseline":
            for player_name in self.player_dict:

                player = self.player_dict[player_name]

                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()

                energy = np.repeat(70.7, len(action)) + np.random.uniform(
                    size=len(action)
                )

                scaler = MinMaxScaler((player_min_demand, player_max_demand))
                energy = scaler.fit_transform(energy.reshape(-1, 1))
                energy = np.squeeze(energy)

                energy_consumptions[player_name] = energy

            energy_consumptions["avg"] = energy

            return energy_consumptions

        else:
            raise ValueError("wrong planning model choice")
            return

    def load_model_from_disk(self, file_name="GPyOpt_planning_model"):
        json_file = open(file_name + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(file_name + ".h5")
        print("Loaded model from disk")

        loaded_model.compile(loss="mse", optimizer="adam")
        return loaded_model

    def step(self, action, step_num=0):
        """
        Purpose: Takes a step in the real environment

        Args:
            Action: 10-dim vector detailing player incentive for each hour (8AM - 5PM)

        Returns:
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        Exceptions:
            raises AssertionError if action is not in the action space
        """

        if not self.action_space.contains(action):
            action = np.asarray(action)
            if self.action_space_string == "continuous":
                action = np.clip(action, 0, 10)

            elif self.action_space_string == "multidiscrete":
                action = np.clip(action, 0, 2)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1

        points = self._points_from_action(action)

        if self.curr_iter > 0:
            done = True
        else:
            done = False

        energy_consumptions = self._simulate_humans(points)

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
        self.prev_energy = energy_consumptions["avg"]

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions)

        info = {}
        return observation, reward, done, info

    def planning_step(self, action, step_num=0):  ## TODO: replace load model in SAC
        """
        Purpose: Takes a step in the environment

        Args:
            Action: 10-dim vector detailing player incentive for each hour (8AM - 5PM)

        Returns:
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        Exceptions:
            raises AssertionError if action is not in the action space
        """

        if not self.action_space.contains(action):
            action = np.asarray(action)
            if self.action_space_string == "continuous":
                action = np.clip(action, 0, 10)

            elif self.action_space_string == "multidiscrete":
                action = np.clip(action, 0, 2)

        prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        # if self.curr_iter > 0:
        #     done = True
        # else:
        #     done = False

        points = self._points_from_action(action)

        if self.curr_iter > 0:
            done = True
        else:
            done = False

        loaded_model = None

        if self.planning_model_type == "LSTM":
            loaded_model = self.load_model_from_disk("GPyOpt_planning_model")

        energy_consumptions = self._planning_prediction(
            action=points,
            day_of_week=self.day_of_week,
            planning_model_type=self.planning_model_type,
            loaded_model=loaded_model,
        )

        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
        self.prev_energy = energy_consumptions["avg"]

        print(energy_consumptions["avg"])

        observation = self._get_observation()
        reward = self._get_reward(prev_price, energy_consumptions)

        info = {}

        return observation, reward, done, info

