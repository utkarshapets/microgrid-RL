import pandas as pd
import numpy as np
import cvxpy as cvx
import scipy.opt
from sklearn.preprocessing import MinMaxScaler
from cvxpy.error import SolverError

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

                try:
                        prob.solve(solver = cvx.SCS)
                except: 
                        try:
                                print("SCS solver did not work")
                                prob.solve(solver = cvx.OSQP)
                        except:
                                try:
                                        print("OSQP or ECOS didn't work")
                                        prob.solve(solver = cvx.ECOS_BB)
                                except:
                                        try:
                                                print("Three didn't work")
                                                prob.solve(solver = cvs.ECOS)
                                        except:
                                                print("none of the solvers work")

                charged = prob.variables()[0].value
                discharged = prob.variables()[1].value
                net = load - gen + charged/eta + discharged*eta

                return np.array(net)

        def get_response_twoprices(
                self, 
                day, 
                buyprice,
                sellprice
                ):

                """
                Determines the net load of the prosumer on a specific day, in response to energy prices
        		
                Args:
        		day: day of the year. Allowed values: [0,365)
        		buyprice: 24 hour price vector, supplied as an np.array
                        sellprice: 24 hour price vector, supplied as an np.array
                        """

                index = day*24+19
                load = self.yearlongdemand[index : index + 24]
                gen = self.pv_size*self.yearlonggeneration[index : index + 24]
                eta = self.eta
                capacity = self.capacity
                battery_num = self.battery_num
                c_rate = self.c_rate
                Ltri = np.tril(np.ones((24, 24)))

                def dailyobjective(x):
                        net = load - gen + (-eta + 1/eta)*abs(x)/2 + (eta + 1/eta)*x/2
                        return sum(np.maximum(net,0)*buyprice) + sum(np.minimum(net,0)*sellprice) 

                def hourly_con_charge_max(x):
                        # Shouldn't charge or discharge too fast
                        return c_rate*capacity*battery_num - x 

                def hourly_con_charge_min(x):
                        # Shouldn't charge or discharge too fast
                        return c_rate*capacity*battery_num + x 

                def hourly_con_cap_max(x):
                        # x should respect the initial state of charge
                        return capacity*battery_num - np.matmul(Ltri,x)

                def hourly_con_cap_min(x):
                        # x should respect the initial state of charge
                        return np.matmul(Ltri,x)

                con1_hourly = {'type':'ineq', 'fun':hourly_con_charge_min}
                con2_hourly = {'type':'ineq', 'fun':hourly_con_charge_max}
                con3_hourly = {'type':'ineq', 'fun':hourly_con_cap_min}
                con4_hourly = {'type':'ineq', 'fun':hourly_con_cap_max}
                cons_hourly = (con1_hourly, con2_hourly, con3_hourly, con4_hourly)
        
                x0 = [battery_num*capacity]*24
                # x0 = [0]*24
                sol = minimize(dailyobjective, x0, constraints=cons_hourly, method='SLSQP', options={'maxiter':10000})
                # net = demand - pv_size*pv_24h + (-eta + 1/eta)*abs(sol['x'])/2 + (eta + 1/eta)*sol['x']/2

                x = sol['x']
                net = demand - pv_size*pv_24h + (-eta + 1/eta)*abs(x)/2 + (eta + 1/eta)*x/2
                # sol['x'] = x
                # sol['fun'] = dailyobjective(x)
                return np.array(net)


        def get_battery_operation(
                self, 
                day, 
                price,
                ):

                """
                Determines the net charge/discharge of the prosumer on a specific day, in response to energy prices
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
        
                try:
                        prob.solve(solver = cvx.SCS)
                except SolverError: 
                        try:
                                prob.solve(solver = cvx.OSQP)
                        except SolverError:
                                try:
                                        prob.solve(solver = cvx.ECOS_BB)
                                except SolverError:
                                        try:
                                                prob.solve(solver = cvs.ECOS)
                                        except SolverError:
                                                print("none of the solvers work")
                        
                charged = prob.variables()[0].value
                discharged = prob.variables()[1].value
                return np.array(charged/eta + discharged*eta)
                
