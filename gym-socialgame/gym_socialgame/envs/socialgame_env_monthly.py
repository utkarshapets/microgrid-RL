import gym
from gym import spaces

import numpy as np

from gym_socialgame.envs.socialgame_env import SocialGameEnv
from gym_socialgame.envs.utils import price_signal
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward

class SocialGameEnvMonthly(SocialGameEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_string = "continuous", response_type_string = "l", number_of_participants = 10,
                one_price = 0, random = False, low = 0, high = 50, distr = 'U', energy_in_state = False, yesterday_in_state = False):
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
        self.check_valid_init_inputs(action_space_string, response_type_string, number_of_participants, one_price, random, low, high, distr,
                                    energy_in_state, yesterday_in_state)
        #Assigning Instance Variables
        self.action_space_string = action_space_string
        self.response_type_string = response_type_string
        self.number_of_participants = number_of_participants
        self.one_price = self._find_one_month(one_price)
        self.energy_in_state = energy_in_state
        self.yesterday_in_state = yesterday_in_state

        #Create Observation Space (aka State Space)
        self.observation_space = self._create_observation_space()
        self.prices = self._get_prices()
        #Day corresponds to day # of the yr
        self.day = 0

        #Cur_iter counts length of trajectory for current step (i.e. cur_iter = i^th hour in a 10-hour trajectory)
        self.cur_iter = 0

        #Tracking reward
        self.reward = 0

        #Create Action Space
        self.action_length = 10
        self.action_subspace = 3
        self.action_space = self._create_action_space()

        #Create Players
        self.random = random
        self.player_dict = self._create_agents()

        #TODO: Check initialization of prev_energy
        self.prev_energy = np.zeros(10)

        print("\n Social Game Monthly Environment Initialized! Have Fun! \n")
    
    
    def _find_one_month(self, one_price: int):
        """
        Purpose: Helper function to find one_price to train on (if applicable)

        Args:
            one_price: (Int) in range [-1,2]

        Returns:
            0 if one_price = 0
            one_price if one_price in range [1,13]
            random_number(1,365) if one_price = -1
        """
        
        if(one_price == -1):
            return np.random.randint(1, high=13)
        
        else:
            return one_price

 
    def _get_prices(self):
        """
        Purpose: Get grid price signals for the entire year (using past data from a building in Los Angeles as reference)

        Args:
            None
            
        Returns: Array containing 365 price signals, where array[day_number] = grid_price for day_number from 8AM - 5PM

        """

        all_prices = []
        if self.one_price != 0:
            # If one_price we repeat the price signals from a fixed month
            # Tweak one_price Price Signal HERE
            month = self.one_price - 1
            for i in range(1, 31):
                price = price_signal(30 * month + i)
                price = np.array(price[8:18])
                price = np.maximum(0.01 * np.ones_like(price), price)
                all_prices.append(price)
    
            all_prices = all_prices * 13 #Doing times 13 just in case, we loop around 365 days so it shouldn't be a concern

        else:
            for day in range(1,366):  
                price = price_signal(day)
                price = np.array(price[8:18])
                # put a floor on the prices so we don't have negative prices
                price = np.maximum(0.01 * np.ones_like(price), price)
                all_prices.append(price)

        
        return all_prices

    

    def step(self, action):
        """
        Purpose: Takes a step in the environment 

        Args:
            Action: 10-dim vector detailing player incentive for each hour (8AM - 5PM)
        
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

        energy_consumptions = self._simulate_humans(points)
        
        # HACK ALERT. USING AVG ENERGY CONSUMPTION FOR STATE SPACE. this will not work if people are not all the same
        self.prev_energy = energy_consumptions["avg"]
        
        #Getting reward
        self.reward += self._get_reward(prev_price, energy_consumptions)

        #Advancing day
        self.day = (self.day + 1) % 365
        self.cur_iter += 1

        #Getting next observation
        observation = self._get_observation()

        #Setting done and return reward
        if self.cur_iter % 30 == 0:
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
        """ Resets the environment to day 0 (of yr or month depending on one_price init) """ 
        #Currently resetting based on current day to work with StableBaselines

        return self._get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass


    def check_valid_init_inputs(self, action_space_string, response_type_string, number_of_participants, one_price, 
                                random, low, high, distr,energy_in_state, yesterday_in_state):
        
        """
        Purpose: Verify that all initialization variables are valid 

        Args (from initialization):
            action_space_string: String either "continuous" or "discrete" ; Denotes the type of action space
            response_type_string: String either "t", "s", "l" , denoting whether the office's response function is threshold, sinusoidal, or linear
            number_of_participants: Int denoting the number of players in the social game (must be > 0 and < 20)
            one_price: Boolean denoting whether (or not) the environment is FIXED on ONE price signal
            random: Boolean denoting whether (or not) to use Domain Randomization
            Low: Int denoting lower bound for random noise
            High: Int denoting upper bound for random noise
            Distr: "G" or "U" denoting "Gaussian" or "Uniform" noise
            energy_in_state: Boolean denoting whether (or not) to include the previous day's energy consumption within the state
            yesterday_in_state: Boolean denoting whether (or not) to append yesterday's price signal to the state

        Exceptions: 
            Raises AssertionError if action_space_string is not a String or if it is not either "continuous", or "discrete"
            Raises AssertionError if response_type_string is not a String or it is is not either "t","s","l"
            Raises AssertionError if number_of_participants is not an integer, is less than 1,  or greater than 20 (upper bound set arbitrarily for comp. purposes).
            Raises AssertionError if any of {one_price, random, energy_in_state, yesterday_in_state} is not a Boolean
            Raises AssertionError if low & high are not integers and low >= high
            Raises AssertionError if distr is not a String and if distr not in ['G', 'U']
        """

        #Checking that action_space_string is valid
        assert isinstance(action_space_string, str), "action_space_str is not of type String. Instead got type {}".format(type(action_space_string))
        action_space_string = action_space_string.lower()
        assert action_space_string in ["continuous", "discrete"], "action_space_str is not continuous or discrete. Instead got value {}".format(action_space_string)

        #Checking that response_type_string is valid
        assert isinstance(response_type_string, str), "Variable response_type_string should be of type String. Instead got type {}".format(type(response_type_string))
        response_type_string = response_type_string.lower()
        assert response_type_string in ["t", "s", "l"], "Variable response_type_string should be either t, s, l. Instead got value {}".format(response_type_string)


        #Checking that number_of_participants is valid 
        assert isinstance(number_of_participants, int), "Variable number_of_participants is not of type Integer. Instead got type {}".format(type(number_of_participants))
        assert number_of_participants > 0, "Variable number_of_participants should be atleast 1, got number_of_participants = {}".format(number_of_participants)
        assert number_of_participants <= 20, "Variable number_of_participants should not be greater than 20, got number_of_participants = {}".format(number_of_participants)

        #Checking that one_price is valid 
        assert isinstance(one_price, int), "Variable one_price is not of type Int. Instead got type {}".format(type(one_price))
        assert 13 > one_price and one_price > -2, "Variable one_price out of range [-1,12]. Got one_price = {}".format(one_price)

        #Checking that energy_in_state is valid
        assert isinstance(energy_in_state, bool), "Variable one_price is not of type Boolean. Instead got type {}".format(type(energy_in_state))

        #Checking that yesterday_in_state is valid
        assert isinstance(yesterday_in_state, bool), "Variable one_price is not of type Boolean. Instead got type {}".format(type(yesterday_in_state))

        #Checking that random and corresp. param are valid
        assert isinstance(random, bool), "Variable random is not of type Boolean. Instead got type {}".format(type(random))
        assert isinstance(low, int), "Variable low is not an integer. Got type {}".format(type(low))
        assert isinstance(high, int), "Variable high is not an integer. Got type {}".format(type(high))
        assert isinstance(distr, str), "Variable distr is not a String. Got type {}".format(type(distr))
        assert distr.upper() in ['G', 'U'], "Distr not either G or U. Got {}".format(distr.upper())