import pandas as pd
import numpy as np
import scipy.stats as stats

def drawdown(return_series: pd.Series):
    """
    Takes a timeseries and returns a DataFrame that includes:
    -The wealth index
    -The previous peaks
    -Percent drawdowns
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French Market Equity dataset for the Top and Bottom Deciles by market cap
    """
    me_monthly = pd.read_csv('./data/Portfolios_Formed_on_ME_monthly_EW.csv',
                            header=0,index_col=0,na_values=-99.99)
   
    returns = me_monthly[['Lo 10', 'Hi 10']]
    returns.columns = ['Small Cap', 'Large Cap']
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format='%Y%m').to_period('M')
    return returns

def drawdown_info(return_series: pd.Series, column):
    """Takes a series and returns a formatted String with the occurance and size of the max drawdown"""
    return 'The max drawdown occured {0!r} and was {1!r}'.format(drawdown(return_series[column])['Drawdown'].idxmin(), drawdown(return_series[column])['Drawdown'].min())

def get_hfi_returns():
    """
    Load and format the EDHEC HEdge Fund Index Returns
    """
    hfi = pd.read_csv('./data/edhec-hedgefundindices.csv',
                            header=0,index_col=0,parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi
  
def semideviation(r):
  """
  Calculates the semideviation (the semideviation of negative returns) of r where r is
  a Series or a DataFrame
  """
  return r[r<0].std(ddof=0)

def var_historic(r, level=5):
  """
  VaR Historic
  When using the numpy array we lose the column information
  """
  # Calls the function again but with every individual Series
  if isinstance(r, pd.DataFrame):
    return r.aggregate(var_historic, level=level)
  elif isinstance(r, pd.Series):
    return -np.percentile(r, level)
  else: 
    raise TypeError("r is expected to be a Series or DataFrame")
    
def cvar_historic(r, level=5):
  """
  CVaR Historic
  Adaptation of var_historic to isolate the returns that are less than the mean
  """
  # Calls the function again but with every individual Series
  if isinstance(r, pd.DataFrame):
    return r.aggregate(cvar_historic, level=level)
  elif isinstance(r, pd.Series):
    is_beyond = r <= -var_historic(r, level)
    return -r[is_beyond].mean()
  else: 
    raise TypeError("r is expected to be a Series or DataFrame")
  
def var_gaussian(r, level=5, modified=False):
  """
  VaR Gaussian
  """
  z = stats.norm.ppf(level/100)
  
  if(modified):
    # Calculate the z score based on observed skewness and kurtosis
    s = skewness(r)
    k = kurtosis(r)
    z = (z + 
          (z**2 - 1)*s/6 + 
          (z**3 -3*z)*(k-3)/24 -
          (2*z**3 - 5*z)*(s**2)/36
        )
  
  return -(r.mean() + z * r.std(ddof=0))

def var_cornish_fisher(r, level=5):
  return var_gaussian(r, level, True)
  
def skewness(returns):
  """
  Manual implementation of scipy.stats.skew()
  Computes the skewness of the supllied series or DataFrame
  returns a float or a Series
  """
  return __skew_kurtosis(returns, 3)
  
def kurtosis(returns):
  """
  Manual implementation of scipy.stats.skew()
  Computes the skewness of the supllied series or DataFrame
  returns a float or a Series
  """
  return __skew_kurtosis(returns, 4)

def __skew_kurtosis(returns, power_to_raise):
  """The equation for skew and kurtosis is the same apart from the power raise by"""
  demeaned_returns = returns - returns.mean()
  # Use the population standard deviation, so set degrees of freedom =0
  sigma_returns = returns.std(ddof=0)
  expected_returns = (demeaned_returns ** power_to_raise).mean()
  return expected_returns / sigma_returns ** power_to_raise

def is_normal(r, level=0.01):
  """
  Applies the Jarque Bera Test
  """
  statistic, p_value = scipy.stats.jarque_bera(r)
  return p_value > level