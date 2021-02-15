import pandas as pd
import numpy as np
import cvxpy as cvx
from sklearn.preprocessing import MinMaxScaler

#### file to make the simulation of prosumers that we can work with 

class Prosumer():
	""" Prosumer (parent?) class -- will define how each prosumer respondes to a price signal to determine its net load
	baseline_energy = a list or dataframe of values. This is net load (load - generation)
	points_multiplier = an int which describes how sensitive each person is to points 

	"""

	def __init__(self, name, yearlongdemand, yearlonggeneration, battery_num = 0, pv_size = 0):

		self.name = name
        self.yearlongdemand = yearlongdemand
		self.yearlonggeneration = yearlonggeneration
        self.battery_num = battery_num
        self.pv_size = pv_size
		self.capacity = 13.5 # kW-hour
		self.batterycyclecost = 273/2800 # per unit capacity
		self.eta = 0.95 #battery one way efficiency
        self.c_rate = 0.35

		# self.baseline_energy_df = baseline_energy_df
		# self.baseline_energy = np.array(self.baseline_energy_df["net_energy_use"])
		# self.points_multiplier = points_multiplier
		
		# baseline_min = self.baseline_energy.min()
		# baseline_max = self.baseline_energy.max()
		# baseline_range = baseline_max - baseline_min
		
		# self.min_demand = np.maximum(0, baseline_min + baseline_range * .05)
		# self.max_demand = np.maximum(0, baseline_min + baseline_range * .95)


	def get_response(self, day, price):
		''' Determines the net load of the prosumer on a specific day, in response to energy prices
		Assumes net metering- single day ahead price set
		Args:
			day: day of the year. Allowed values: [1,364]
			price: 24 hour price vector, supplied as an np.array
		'''
		index = day*24-5
        load = self.yearlongdemand[index: index+24]
        gen = self.pv_size*self.yearlonggeneration[index:index+24]
        eta = self.eta
        Ltri = np.tril(np.ones((24, 24)))

        charge = cvx.Variable(24) # positive
        discharge = cvx.Variable(24) # negative
        
        # obj = cvx.Minimize(price.T@(load - gen + charge/eta + discharge*eta) + self.batterycyclecost*(sum(charge)))
        obj = cvx.Minimize(price.T@(load - gen + charge/eta + discharge*eta))
        constraints = [Ltri@(charge + discharge) <= self.capacity*self.battery_num*np.ones(24),
                        Ltri@(charge + discharge) >= np.zeros(24),
                        charge >= np.zeros(24),
                        charge <= self.c_rate*self.capacity*self.battery_num*np.ones(24),
                        discharge >= -self.c_rate*self.capacity*self.battery_num*np.ones(24),
                        discharge <= np.zeros(24)]
        prob = cvx.Problem(obj, constraints)
        
        prob.solve()
        
        charged = prob.variables()[0].value
        discharged = prob.variables()[1].value
        net = load - gen + charged/eta + discharged*eta
        return np.array(net)

