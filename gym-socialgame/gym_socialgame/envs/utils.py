import csv
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 

#data_path = os.path.join(os.getcwd(), "baselines", "behavioral_sim", "building_data.csv")
# csv_path = os.path.dirname(os.path.realpath(__file__)) + "/building_data.csv"


def price_signal(day = 45, type_of_DR = "real_time_pricing"):

    """
    Utkarsha's work on price signal from a building with demand and solar
    Input: Day = an int signifying a 24 hour period. 365 total, all of 2012, start at 1.
    Output: netdemand_price, a measure of how expensive energy is at each time in the day
        optionally, we can return the optimized demand, which is the building
        calculating where the net demand should be allocated
    """
    csv_path = "building_data.csv"
    csv_path_2 = "../gym-socialgame/gym_socialgame/envs/building_data.csv"
    csv_path_3 = "/global/home/users/lucas_spangher/transactive_control/gym-socialgame/gym_socialgame/envs/building_data.csv"
    try:
        df = pd.read_csv(csv_path)
    except:
        try:
            df = pd.read_csv(csv_path_2)
        except:
            df = pd.read_csv(csv_path_3)

    pv = 0.001*np.array(df['PV (W)'].tolist())
    price = np.array(df['Price( $ per kWh)'].tolist())
    demand = np.nan_to_num(np.array(df['Office_Elizabeth (kWh)'].tolist()))
    demand_charge = 10/30 # 10$/kW per month

    pvsize = 0 #Assumption
    netdemand = demand - pvsize*pv

    # Data starts at 5 am on Jan 1
    netdemand_24 = netdemand[24*day-5:24*day+19]
    price_24 = price[24*day-5:24*day+19]
    pv_24 = pv[24*day-5:24*day+19]
    demand_24 = demand[24*day-5:24*day+19]

    # Calculate optimal load scheduling. 90% of load is fixed, 10% is controllable.
    def optimise_24h(netdemand_24, price_24):
        currentcost = netdemand_24@price_24

        fixed_load = 0.9*netdemand_24
        controllable_load = sum(0.1*netdemand_24)
        
        def objective(x):
            load = fixed_load + x
            cost = np.multiply(price_24,load)
            # Negative demand means zero cost, not negative cost
            # Adding L1 regularisation to penalise shifting of occupant demand
            lambd = 0.005
            # Demand charge: attached to peak demand
            return sum(np.maximum(cost,0)) + demand_charge*max(load) + lambd*sum(abs(x-0.1*netdemand_24))

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

    if type_of_DR == "real_time_pricing":    
        sol = optimise_24h(netdemand_24,price_24)
        x = sol['x']
        diff = x - 0.1*netdemand_24
        return -diff - min(-diff)

    elif type_of_DR == "time_of_use":
        if (np.mean(price_24[8:18]) == price_24[9]):
            price_24[13:16]+=3
        return price_24
    else:
        return "error!!!"















