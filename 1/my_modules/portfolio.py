import numpy as np
import pandas as pd

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

