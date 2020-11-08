import gym
from gym import spaces

import numpy as np

from gym_socialgame.envs.socialgame_env import SocialGameEnv
from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward

class SocialGameEnvHourly(SocialGameEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_string = "continuous", response_type_string = "l", number_of_participants = 10,
                one_price = 0, random = False, energy_in_state = False, yesterday_in_state = False):
        """
        SocialGameEnv for an agent determining incentives in a social game. 
        
        Note: One-step trajectory (i.e. agent submits a 10-dim vector containing incentives for each hour (8AM - 5PM) each day. 
            Then, environment advances one-day and agent is told that the episode has finished.)

        Args:
            action_space_string: (String) either "continuous", or "multidiscrete"
            response_type_string: (String) either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: (Int) denoting the number of players in the social game (must be > 0 and < 20)
            one_price: (Int) in range [-1,365] denoting which fixed day to train on . 
                    Note: -1 = Random Day, 0 = Train over entire Yr, [1,365] = Day of the Year
            Random: (Boolean) denoting whether or not to use Domain Randomization
            energy_in_state: (Boolean) denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: (Boolean) denoting whether (or not) to append yesterday's price signal to the state

        """
        
        #Verify that inputs are valid 
        self.check_valid_init_inputs(action_space_string, response_type_string, number_of_participants, one_price, energy_in_state, yesterday_in_state)

        #Assigning Instance Variables
        self.action_space_string = action_space_string
        
        self.response_type_string = response_type_string
        self.number_of_participants = number_of_participants
        self.one_price = self._find_one_day(one_price)
        self.energy_in_state = energy_in_state
        self.yesterday_in_state = yesterday_in_state

        #Create Observation Space (aka State Space)
        self.observation_space = self._create_observation_space()
        self.prices = self._get_prices()
        #Day corresponds to day # of the yr
        self.day = 0

        #Hour corresponds to Hour of day (setting b/c cur_iter is unbounded basically it counts the # of steps)
        #Note self.hour is in [0,10] which maps to -> 8AM to 5PM
        self.hour = 0

        #Tracking reward
        self.reward = 0

        #Create Action Space
        self.action_length = 1
        self.action_space = self._create_action_space()

        #Create Players
        self.random = random
        self.player_dict = self._create_agents()

        #TODO: Check initialization of prev_energy
        self.prev_energy = np.zeros(1)


        print("\n Social Game Hourly Environment Initialized! Have Fun! \n")
    
    def _create_observation_space(self):
        """
        Purpose: Returns the observation space

        Args:
            None

        Returns:
            Space based on yesterday_in_state, and energy_in_state obj param
        """

        if(self.yesterday_in_state):
            if(self.energy_in_state):
                return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            else:
                return spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        else:
            if self.energy_in_state:
                return spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            else:
                return spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    
    def _create_action_space(self):
        """
        Purpose: Return action space of type specified by self.action_space_string

        Args:
            None
        
        Returns:
            Action Space for environment based on action_space_str 
        
        Note: Discrete refers to a values [0,10] 
        """

        #TODO: Create {Low, Med, High} Actions


        #Making a symmetric, continuous space to help learning for continuous control (suggested in StableBaselines doc.)
        if self.action_space_string == "continuous":
            return spaces.Box(low=-1, high=1, shape=(self.action_length,), dtype=np.float32)

        elif self.action_space_string == "discrete":
            return spaces.Discrete(10)

    def _points_from_action(self, action):
        """
        Purpose: Convert agent actions into incentives (conversion is for multidiscrete setting)

        Args:
            Action: 10-dim vector corresponding to action for each hour 8AM - 5PM
        
        Returns: Points: 10-dim vector of incentives for game (same incentive for each player)
        """
        if self.action_space_string == "discrete":
            #Mapping 0 -> 0.0, 1 -> 5.0, 2 -> 10.0
            points = action

        elif self.action_space_string == 'continuous':
            #Continuous space is symmetric [-1,1], we map to -> [0,10] by adding 1 and multiplying by 5
            points = 5 * (action + np.ones_like(action))
        
        return points

    def _get_observation(self):
        """ Returns observation for current hour """ 
        
        #Observations are per hour now
        prev_price = np.array([self.prices[self.day][(self.hour - 1) % 10]])
        next_observation = np.array([self.prices[self.day][self.hour]])

        if(self.yesterday_in_state):
            if self.energy_in_state:
                return np.concatenate((next_observation, np.concatenate((prev_price, self.prev_energy))))
            else:
                return np.concatenate((next_observation, prev_price))

        elif self.energy_in_state:
            return np.concatenate((next_observation, self.prev_energy))

        else:
            return next_observation


    def step(self, action):
        """
        Purpose: Takes a step in the environment 

        Args:
            Action: 1-dim vector detailing player incentive for a given hour
        
        Returns: 
            Observation: State for the next day
            Reward: Reward for said action
            Done: Whether or not the day is done (should always be True b/c of 1-step trajectory)
            Info: Other info (primarily for gym env based library compatibility)

        """
        #Checking that action is valid; If not, we clip (OpenAI algos don't take into account action space limits so we must do it ourselves)
        if(not self.action_space.contains(action)):
            action = np.asarray(action)
            if(self.action_space_string == 'continuous'):
                action = np.clip(action, 0, 10)

            elif(self.action_space_string == 'multidiscrete'):
                action = np.clip(action, 0, 2) 

        prev_price = self.prices[self.day]

        points = self._points_from_action(action)

        #TODO: FIX ENERGY CONSUMPTION (Player consumption must be vectorized!)
        energy_consumptions = self._simulate_humans(points)
        
        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
        self.prev_energy = energy_consumptions["avg"]
        
        #Getting reward
        self.reward += self._get_reward(prev_price, energy_consumptions)

        #Advancing hour
        self.hour += 1

        #Getting next observation
        observation = self._get_observation()

        #Setting done and return reward
        if self.hour == 10:
            #Reset hour
            self.hour = 0

            #Advance one day
            self.day = (self.day + 1) % 365

            #Finish episode
            done = True
            reward = self.reward
            self._update_randomization()

        else:
            done = False
            reward = 0.0
        
        #Setting info for baselines compatibility (no relevant info for us)
        info = {}

        return observation, reward, done, info


    #Keeping reset, render, close for clarity sake
    def reset(self):
        """ Resets the environment on the current day """ 
        #Currently resetting based on current day to work with StableBaselines

        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
