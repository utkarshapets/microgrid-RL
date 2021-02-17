import pandas as pd
import numpy as np
import cvxpy as cvx
from sklearn.preprocessing import MinMaxScaler

#### file to make the simulation of prosumers that we can work with 

class Prosumer():
        """ 
        Prosumer (parent?) class -- will define how each prosumer respondes to a price signal to determine its net load
	baseline_energy = a list or dataframe of values. This is net load (load - generation)
	points_multiplier = an int which describes how sensitive each person is to points 

	"""	
        def __init__(
                self, 
                name, 
                yearlongdemand, 
                yearlonggeneration, 
                battery_num = 0, 
                pv_size = 0,
                ):
                
                self.name = name
                self.yearlongdemand = yearlongdemand
                self.yearlonggeneration = yearlonggeneration
                self.battery_num = battery_num
                self.pv_size = pv_size
                self.capacity = 13.5 # kW-hour
                self.batterycyclecost = 273/2800 # per unit capacity
                self.eta = 0.95 #battery one way efficiency
                self.c_rate = 0.35

        def get_response(
                self, 
                day, 
                price,
                ):

                """
                Determines the net load of the prosumer on a specific day, in response to energy prices
        	Assumes net metering- single day ahead price set
        		
                Args:
        		day: day of the year. Allowed values: [0,365)
        		price: 24 hour price vector, supplied as an np.array
                        """

                index = day*24+19
                load = self.yearlongdemand[index : index + 24]
                gen = self.pv_size*self.yearlonggeneration[index : index + 24]
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
        


        
                charged = prob.variables()[0].value
                discharged = prob.variables()[1].value
                net = load - gen + charged/eta + discharged*eta

                return np.array(net)

