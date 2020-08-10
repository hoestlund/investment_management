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
  
def msrp(riskfree_rate, er, cov):
    """
    Riskfree rate + ER + Cov -> W
    Returns maximum sharpe rate portfolio weightings given expected and a covariance matrix
    """
    num_assets = er.shape[0]
    init_guess = np.repeat(1/num_assets, num_assets)
    
    # Constraint 1: Bounds, not leveraged or short
    bounds = ((0.0, 1.0),) * num_assets
    
    # Constraint 2: weights sum to 1
    weight_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # a bit backwards, we want max sharpe which is the smallest neg_sharpe
    results = minimize(__neg_sharpe_ratio, init_guess,
                        args=(riskfree_rate, er, cov), method='SLSQP',
                        options={'disp': False},
                        constraints=(weight_sum_1),
                        bounds=bounds
                      )
    return results.x
  
def __neg_sharpe_ratio(weights, riskfree_rate, er, cov):
  """
  Returns the negative of the sharpe ratio, given weights
  """
  ret = returns(weights, er)
  vol = volatility(weights, cov)
  return -(ret - riskfree_rate)/ vol

def plot_n_asset_frontier(num_points, ex_return, cov, show_cml=False, riskfree_rate=0, style='.-'):
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
                          
  ax = frontier.plot.line(x='Volatility', y='Returns', style=style)
  if(show_cml):
    ax.set_xlim(left=0)
    w_msr = msrp(riskfree_rate, ex_return, cov)
    r_msr = returns(w_msr, ex_return)
    vol_msr = volatility(w_msr, cov)
    # Add Capital Market Line
    cml_x = [0, vol_msr] # We know two points of x
    cml_y = [riskfree_rate, r_msr]
    ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', markersize=9, linewidth=2)
  return ax
  