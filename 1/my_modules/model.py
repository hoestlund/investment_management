import numpy as np
import pandas as pd
import math

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Function from course resources.
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val
  
def inst_to_ann(r):
    """
    Converts short rate to an annualised rate
    """
    return np.expm1(r) # == np.exp(r) - 1

def ann_to_inst(r):
    """
    Converts annualised rate to short rate
    """
    return np.log1p(r) # == np.log(1 + r)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Gives the random interest rate eveolution over time using the Cox Ingersoll Ross model.
    b and r_0 are assumed to be the annaulised (not short) rate, with the return values being given as annualised rates too.
    """
    if r_0 is None: r_0=b
    r_0 = ann_to_inst(r_0) #For small values of r the instantaneosu rate is not that different to the annual
    dt = 1/steps_per_year
    num_steps = int(n_years * steps_per_year) + 1 # One added as n_years might be given as a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## For price generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    
    prices[0] = price(n_years, r_0)
    
    for step in range(1, num_steps):
        rt = rates[step - 1]
        d_r_t = a * (b - rt) + sigma*np.sqrt(rt)*shock[step]
        rates[step] = abs(rt + d_r_t) #This should be a positive number but to be safe with rouding errors/high shocks abs
        # Generate a price at time t as well
        prices[step] = price(n_years-step*dt, rates[step])
        
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    
    return rates, prices