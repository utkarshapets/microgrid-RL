from agents import Person, FixedDemandPerson, DeterministicFunctionPerson
from reward import Reward
import controller
from controller import BaseController, PGController, SimpleNet
import pandas as pd
from utils import *
from dataloader import *
import csv
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.pyplot as plt
import IPython
import datetime
import os
import torch


class Office():
	def __init__(
		self,
		iterations = 1000,
		transfer=False, 
		nn_filepath_transfer = None, 
		opt_filepath_transfer = None,
		nn_file_to_name = None,
		opt_file_to_name = None):

		self._start_timestamp = pd.Timestamp(year=2012,
                                         month=1,
                                         day=2,
                                         hour=0,
                                         minute=0)
		self._end_timestamp = pd.Timestamp(year=2012,
                                         month=12,
                                         day=30,
                                         hour=0,
                                         minute=0)
		self._end_timestamp = pd.Timestamp(year=2012,
                                           month=12,
                                           day=30,
                                           hour=0,
                                           minute=0)
		self._timestep= self._start_timestamp
		self._time_interval = timedelta(days=1)
		self.players_dict = self._create_agents()

		if not transfer:
			self.controller = self._create_controller()
		else: 
			self.controller = self._create_controller(
				transfer=True, 
				nn_filepath=nn_filepath_transfer, 
				opt_filepath = opt_filepath_transfer)

		self.num_iters = iterations
		self.current_iter = 0

		filename = str(datetime.date.today()) + ".txt"
		self.log_file = os.path.join( "simulation_logs/" + nn_file_to_name + filename)
		
		nn_date = str(datetime.date.today()) + ".pth"
		self.nn_file = os.path.join( "nn_logs/" + nn_file_to_name + nn_date)

		opt_date = str(datetime.date.today()) + ".pth"
		self.opt_file = os.path.join( "opt_logs/" + opt_file_to_name + opt_date)

	def _create_agents(self):
		"""Initialize the market agents
			Args:
			  None

			Return:
			  agent_dict: dictionary of the agents
        """

		print("creating agents")

		#Skipping rows b/c data is converted to PST, which is 16hours behind
		# so first 10 hours are actually 7/29 instead of 7/30
		
		# baseline_energy1 = convert_times(pd.read_csv("wg1.txt", sep = "\t", skiprows=range(1, 41)))
		# baseline_energy2 = convert_times(pd.read_csv("wg2.txt", sep = "\t", skiprows=range(1, 41)))
		# baseline_energy3 = convert_times(pd.read_csv("wg3.txt", sep = "\t", skiprows=range(1, 41)))

		# be1 = change_wg_to_diff(baseline_energy1)
		# be2 = change_wg_to_diff(baseline_energy2)
		# be3 = change_wg_to_diff(baseline_energy3)

		players_dict = {}

		# I dont trust the data at all
		# helper comment         [0, 1, 2, 3, 4, 5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 19, 20,  21, 22, 23]
		sample_energy = np.array([0, 0, 0, 0, 0, 0, 20, 50, 80, 120, 200, 210, 180, 250, 380, 310, 220, 140, 100, 50, 20,  10,  0,  0])
		my_baseline_energy = pd.DataFrame(data={"net_energy_use": sample_energy})


		players_dict['player_0'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_1'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_2'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_3'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_4'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_5'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_6'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)
		players_dict['player_7'] = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 100)

		return players_dict

	def _create_controller(self, transfer = False, nn_filepath = None, opt_filepath = None):
		print("creating controller")
		# controller initialize -- hyperparameters
		# different types of controllers, and down the line, pick the one we use.
		# controller.initialize(hyperparameters = hyperparameters)

		if transfer:
			model = SimpleNet()
			model.load_state_dict(torch.load(nn_filepath))
			opt = torch.load(opt_filepath)
			controller = PGController(policy = model, transfer = 3)

			# controller.optimizer.load_state_dict(torch.load(opt_filepath))
		
		else:
			controller = PGController()

		return controller

	def get_timestep(self):
		return self._timestep

	def step(self, prices):
		"""
		- get what the controller would output
		- controller.update to pass in reward
		- controller initiatlization
		"""

		# get controllers points
		controller = self.controller
		controllers_points = controller.get_points(prices)

		end = False

		energy_dict = {}
		rewards_dict = {}
		for player_name in self.players_dict:

			# get the points output from players
			player = self.players_dict.get(player_name)
			player_energy = player.threshold_exp_response(controllers_points.numpy())
			last_player_energy = player_energy
			energy_dict[player_name] = player_energy

			# get the reward from the player's output
			player_min_demand = player.get_min_demand()
			player_max_demand = player.get_max_demand()
			player_reward = Reward(player_energy, prices, player_min_demand, player_max_demand)
			player_ideal_demands = player_reward.ideal_use_calculation()
			last_player_ideal = player_ideal_demands
			# either distance from ideal or cost distance
			# distance = player_reward.neg_distance_from_ideal(player_ideal_demands)

			# print("Ideal demands: ", player_ideal_demands)
			# print("Actual demands: ", player_energy)
			reward = player_reward.scaled_cost_distance_neg(player_ideal_demands)
			rewards_dict[player_name] = reward

		total_reward = sum(rewards_dict.values())

		# reward goes back into controller as controller update

		controller.update(total_reward, prices, controllers_points)

		self._timestep = self._timestep + self._time_interval

		if self._timestep>self._end_timestamp:
			self._timestep = self._start_timestamp

		if self.current_iter >= self.num_iters:
			end = True

		self.current_iter += 1
		return controllers_points, last_player_energy, last_player_ideal, total_reward, end

	def log(self, reward, actions, price_signal, demands, ideal_demands):
		row = {}
		row["iteration"] = self.current_iter
		row["timestamp"] = self._timestep
		row["demands"] = demands
		row["ideal_demands"] = ideal_demands
		row["reward"] = reward
		row["actions"] = actions
		row["price_signal"] = price_signal
		df = pd.DataFrame(data=row)
		df.index.name = "hour"
		with open(self.log_file, "a") as f:
			if row["iteration"] == 1:
				df.to_csv(f, header=True)
			else:
				df.to_csv(f, header=False)


	def price_signal(self, day = 45):

		"""
		Utkarsha's work on price signal from a building with demand and solar

		Input: Day = an int signifying a 24 hour period. 365 total, all of 2012, start at 1.
		Output: netdemand_price, a measure of how expensive energy is at each time in the day
			optionally, we can return the optimized demand, which is the building
			calculating where the net demand should be allocated
		"""

		pv = np.array([])
		price = np.array([])
		demand = np.array([])

		with open('building_data.csv', encoding='utf8') as csvfile:
		    csvreader = csv.reader(csvfile, delimiter=',')
		    next(csvreader,None)
		    rowcount = 0
		    for row in csvreader:
		        pv = np.append(pv, 0.001*float(row[3])) # Converting Wh to kWh
		        price = np.append(price, float(row[2])) # Cost per kWh
		        val = row[5]
		        if val in (None,""): #How to treat missing values
		            val = 0
		        else:
		            val = float(val) # kWh
		        demand = np.append(demand, val)
		        rowcount+=1
		        # if rowcount>100:
		        #     break

		pvsize = 5 #Assumption

		netdemand = demand.copy()
		for i in range(len(demand)):
		    netdemand[i] = demand[i] - pvsize*pv[i]

		# Data starts at 5 am on Jan 1
		netdemand_24 = netdemand[24*day-5:24*day+19]
		price_24 = price[24*day-5:24*day+19]
		pv_24 = pv[24*day-5:24*day+19]
		demand_24 = demand[24*day-5:24*day+19]

		# Calculate optimal load scheduling. 90% of load is fixed, 10% is controllable.
		def optimise_24h(netdemand_24, price_24):
		    currentcost = netdemand_24*price_24

		    fixed_load = 0.9*netdemand_24
		    controllable_load = sum(0.1*netdemand_24)
		    # fixed_load = 0*netdemand_24
		    # controllable_load = sum(netdemand_24)

		    def objective(x):
		        load = fixed_load + x
		        cost = np.multiply(price_24,load)
		        # Negative demand means zero cost, not negative cost
		        # Adding L1 regularisation to penalise shifting of occupant demand
		        lambd = 0.005
		        return sum(np.maximum(cost,0)) + lambd*sum(abs(x-0.1*netdemand_24))

		    def constraint_sumofx(x):
		        return sum(x) - controllable_load

		    def constraint_x_positive(x):
		        return x

		    x0 = np.zeros(24)
		    cons = [
		        {'type':'eq', 'fun': constraint_sumofx},
		        {'type':'ineq', 'fun':constraint_x_positive}
		    ]
		    sol = minimize(objective, x0, constraints=cons)
		    return sol

		sol = optimise_24h(netdemand_24,price_24)
		x = sol['x']

		netdemand_price_24 = netdemand_24*price_24

		return(netdemand_price_24)

def main():
	prefix = "base_sday_threshexp_with_exp_trained_1_"
	# prefix = "base_sday_linear_"
	test_office = Office(
		iterations=2000,
		transfer = True, 
		nn_filepath_transfer = "nn_logs/nn_logs_base_sday_exp_1_2019-12-16.pth", 
		opt_filepath_transfer= "opt_logs/opt_logs_base_sday_exp_1_2019-12-16.pth",
		nn_file_to_name= "nn_logs_" + prefix,
		opt_file_to_name= "opt_logs_" + prefix)
	end = False
	rewards = []
	day = 1
	point_curves = []
	total_iterations = 0
	log_frequency = 1
	f = open(test_office.log_file, "w+")
	f.close()

	with open("temp_reward_values.txt", "w") as f:
		while not end:
			timestep = test_office.get_timestep()
			print("--------Iteration: " + str(total_iterations) + " Timestep: " + str(timestep) + "-------")

			# ALWAYS SAME DAY FOR TESTING
			prices = test_office.price_signal(20)
			points, last_demand, last_player_ideal, reward, end = test_office.step(prices)
			print("Controller Action: ", points)
			print("Reward: ", reward)
			day = ((day + 1) % 365) + 1
			total_iterations += 1
			if day % 1000 == 1:
				point_curves.append(points)
			rewards.append(reward)

			if day % log_frequency == 0:
				test_office.log(
					reward = reward, 
					actions = points, 
					price_signal = prices, 
					demands = last_demand, 
					ideal_demands = last_player_ideal
					)

			f.write(str(reward) + "\n")
			f.flush()

			if total_iterations % 100 ==0:

				## save neural net parameters every 100 iterations
				with open(test_office.nn_file,"wb") as nn:
					torch.save(test_office.controller.policy_net.state_dict(), nn)

				with open(test_office.opt_file, "wb") as opt_file:
					torch.save(test_office.controller.optimizer.state_dict(), opt_file)

		plt.plot(rewards)
		plt.title("Same day, base, thresh, 2 hidden nodes")
		plt.show()

		for i, curve in enumerate(point_curves):
			plt.figure()
			plt.plot(curve, label="curve " + str(i))
		plt.legend()
		plt.show()



if __name__ == "__main__":
	main()
