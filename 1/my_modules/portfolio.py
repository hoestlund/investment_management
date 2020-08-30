import numpy as np
import pandas as pd
from scipy.optimize import minimize
from my_modules import risk

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
  
def cppi(risky_r, safe_r=None, start_value=1000, floor=0.8, m=3, riskfree_rate=0.03, drawdown=None):
  """
  Returns the backtest of a CPPI allocation strategy given a static floor level.
  Floor value given as a percentage of the start_value in form of 0.x
  If drawdown (constraint) the floor value is no longer static
  """
  dates = risky_r.index
  n_steps = len(dates)
  account_value = start_value
  floor_value = start_value * floor
  peak = start_value
  
  if isinstance(risky_r, pd.Series):
    risky_r = pd.DataFrame(risky_r, columns=['R'])
    
  if safe_r is None:
    safe_r = pd.DataFrame().reindex_like(risky_r)
    safe_r.values[:] = riskfree_rate / 12
    
  # Save backtest, running the algorithm across past values
  account_history = pd.DataFrame().reindex_like(risky_r)
  cushion_history = pd.DataFrame().reindex_like(risky_r)
  risky_weight_history = pd.DataFrame().reindex_like(risky_r)

  for step in range(n_steps):
    if drawdown is not None:
      peak = np.maximum(peak, account_value)
      floor_value = peak*(1-drawdown)
    cushion = (account_value - floor_value)/account_value
    risky_w = m*cushion
    # Don't lever or go short
    risky_w = np.minimum(risky_w, 1)
    risky_w = np.maximum(risky_w, 0)
    safe_w = 1 - risky_w
    risky_alloc = account_value * risky_w
    safe_alloc = account_value * safe_w
    ## Update account value for t
    account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
    ## Save values to plot history
    account_history.iloc[step] = account_value
    cushion_history.iloc[step] = cushion
    risky_weight_history.iloc[step] = risky_w
    
  risky_wealth = start_value*(1+risky_r).cumprod()
    
  backtest_result = {
    "Wealth" : account_history,
    "Risky Wealth" : risky_wealth,
    "Risk Budget" : cushion_history,
    "Risky Allocation" : risky_weight_history,
    "m" : m,
    "Start" : start_value,
    "Floor" : floor,
    "risky_r" : risky_r,
    "safe_r" : safe_r
  }
  
  return backtest_result
  
  
def __neg_sharpe_ratio(weights, riskfree_rate, er, cov):
  """
  Returns the negative of the sharpe ratio, given weights
  """
  ret = returns(weights, er)
  vol = volatility(weights, cov)
  return -(ret - riskfree_rate)/ vol

def gmv(cov):
  """
  Returns weights of the global minimum portfolio
  """
  # Calculating the Sharpe Ratio when returns are the same the only thing to optimise is the volality
  n = cov.shape[0]
  return msrp(0, np.repeat(1, n), cov)

def plot_n_asset_frontier(num_points, ex_return, cov, show_cml=False, show_ew=False, show_gmv=False, riskfree_rate=0, style='.-'):
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
  if(show_ew):
    n = ex_return.shape[0]
    w_ew = np.repeat(1/n, n)
    r_ew = returns(w_ew, ex_return)
    v_ew = volatility(w_ew, cov)
    ax.plot([v_ew], [r_ew], color='goldenrod', marker='o', markersize=9)
  if(show_gmv):
    w_gmv = gmv(cov)
    r_gmv = returns(w_gmv, ex_return)
    v_gmv = volatility(w_gmv, cov)
    ax.plot([v_gmv], [r_gmv], color='limegreen', marker='*', markersize=16)
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

def summary_stats(r, riskfree_rate=0.03):
  """
  DataFrame of summary statistics
  """
  ann_r = r.aggregate(risk.annualise_rets)
  ann_vol = r.aggregate(risk.annualise_vol)
  ann_sr = r.aggregate(risk.sharpe_ratio, risk_free_rate=riskfree_rate)
  dd = r.aggregate(lambda r: risk.drawdown(r).Drawdown.min())
  skew = r.aggregate(risk.skewness)
  kurtosis = r.aggregate(risk.kurtosis)
  cf_var5 = r.aggregate(risk.var_cornish_fisher)
  hist_cvar = r.aggregate(risk.cvar_historic)
  
  return pd.DataFrame({
    "Annualised Return" : ann_r,
    "Annulised Volatility" : ann_vol,
    "Max Drawdown" : dd,
    "Skew" : skew,
    "Kurtosis" : kurtosis,
    "Cornish-Fisher VaR (5%)" : cf_var5,
    "Historic CVaR (5%)" : hist_cvar,
    "Annualised Sharpe Ration" : ann_sr
  })
def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t, given annual interest rate r.
    Assumes that the yield curve is flat (duration does not matter).
    Returns a  |t| x |r| Series or DataFrame (df indexed by t)
    r can be a float, Series, or DataFrame
    """
    discounts = pd.DataFrame([(1 + r)**-i for i in t])
    discounts.index = t
    return  discounts
  
def pv(flows,r):
    """
    Compute the present value of a sequence of cash flows given by the time (index) and amounts r can be scalar, Series, or DataFrame with the number of rows matching the nums of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes funding ratio given a set of assets, liabilites and a constant interest rate where l is a series of liabilites with the time until maturity
    """
    return pv(assets,r)/pv(liabilities, r)
  
def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
  
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()
  
def macaulay_duration(cash_flows, discount_rate):
    """
    Computes the Macaulay Duration of the bond based on the cash flows and the discount rate
    """
    discounted_flows = discount(cash_flows.index, discount_rate) * cash_flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(discounted_flows.index,weights=weights)
  
def match_durations(cashflows_target, cashflows_shortbond, cashflows_longbond, discount_rate):
    """
    Find the weight of the short bond so that, together with 1 - short bond weight, there will be an duration
    that matches the cashflows of the target
    """
    d_target = macaulay_duration(cashflows_target, discount_rate)
    d_short = macaulay_duration(cashflows_shortbond, discount_rate)
    d_long = macaulay_duration(cashflows_longbond, discount_rate)
    
    return (d_long - d_target) / (d_long - d_short)