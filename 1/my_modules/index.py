from my_modules import risk

def total_marketcap(industry_sizes, num_firms):
  """
  Returns the marketcap across all indistries for each given period of data
  """
  if(industry_sizes.shape != num_firms.shape):
    raise ValueError('The shape of the data for industry sizes and number of firms need to be the same')
  
  ind_mktcap = industry_sizes * num_firms
  return ind_mktcap.sum(axis='columns')

def component_capweight(industry_sizes, num_firms):
  """
  Returns the weightings of each component of the market over a period of time
  """
  ind_mktcap = industry_sizes * num_firms
  total_mktcap = total_marketcap(industry_sizes, num_firms)
  return ind_mktcap.divide(total_mktcap, axis='rows')

def total_market_return(industry_sizes, num_firms, industry_returns):
  """
  Calculates the total market return with the weightings of the components taken into account
  """
  if(industry_returns.shape != industry_sizes.shape or industry_returns.shape != num_firms.shape):
    raise ValueError('The returns data needs to have the same shapes as the industry and firm numbers data')
  capweights = component_capweight(industry_sizes, num_firms)
  return (capweights * industry_returns).sum(axis='columns')

def total_market_index(industry_sizes, num_firms, industry_returns):
  TMR = total_market_return(industry_sizes, num_firms, industry_returns)
  return risk.drawdown(TMR).Wealth