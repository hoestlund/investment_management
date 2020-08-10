import numpy as np
import pandas as pd
from scipy.optimize import minimize

def returns(weights, returns):
  """
  Weights -> Returns
  """
  # Transposing weights and doing matrix multiplication with the returns
  return weights.T @ returns

def volatility(weights, covmat):
  """
  Weights -> Vol
  """
  return (weights.T @ covmat @ weights) ** 0.5

def plot_two_asset_frontier(num_points, ex_return, cov):
  """
  Plots the efficient frontier for a two asset mix
  """
  if ex_return.shape[0] != 2 or cov.shape[0] != 2:
    raise ValueError('Only two assets can be used')
    
  weights = [np.array([w, 1-w]) for w in np.linspace(0,1,num_points)]
  ret = [returns(w, ex_return) for w in weights]
  vol = [volatility(w, cov) for w in weights]
  frontier = pd.DataFrame({
    'Returns':ret,
    'Volatility':vol
  })
  return frontier.plot.line(x='Volatility', y='Returns', style='.-')

def minimise_vol(target_return, er, cov):
    """
    target_return -> weight vector (W)
    """
    num_assets = er.shape[0]
    init_guess = np.repeat(1/num_assets, num_assets)
    
    # Constraint 1:
    # need a sequence of bounds for every weight vector
    # it is constrained to not use leverage or go short
    bounds = ((0.0, 1.0),) * num_assets # n tuples of a tuple
    
    # Constraint 2:
    # The portfolio for the target_return should have the lowest volatility
    return_is_target = {
        'type': 'eq', #equality
        'args': (er,),
        'fun': lambda weights, er: target_return - returns(weights,er)
    }
    
    # Constraint 3:
    # The weights sum to 1
    weight_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(volatility, init_guess,
                      args=(cov,), method='SLSQP',
                      options={'disp': False},
                       constraints=(return_is_target, weight_sum_1),
                       bounds=bounds
                      )
    return results.x
  
def optimal_weights(n_points, ex_return, cov):
    # need to generate a list of target returns to send to optimiser
    target_rs = np.linspace(ex_return.min(), ex_return.max())
    weights = [minimise_vol(target_rt, ex_return, cov) for target_rt in target_rs]
    return weights

def plot_n_asset_frontier(num_points, ex_return, cov):
  """
  Plots the efficient frontier for an n asset mix
  """
  weights = optimal_weights(num_points, ex_return, cov)
  rets = [returns(w, ex_return) for w in weights]
  vol = [volatility(w, cov) for w in weights]
  frontier = pd.DataFrame({
    'Returns':rets,
    'Volatility':vol
  })
  return frontier.plot.line(x='Volatility', y='Returns', style='.-')

