import numpy as np
import pandas as pd

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0 = 100.0):
    """Evolution of a stock price using a geomertric Brownian mition model"""
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)
    rets_plus_one = np.random.normal(loc=(1 + mu*dt),scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_one[0] = 1
    prices = s_0 * pd.DataFrame(rets_plus_one).cumprod()
    return prices