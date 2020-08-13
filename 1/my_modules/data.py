import pandas as pd

def get_hfi_returns():
    """
    Load and format the EDHEC HEdge Fund Index Returns
    """
    hfi = pd.read_csv('./data/edhec-hedgefundindices.csv',
                            header=0,index_col=0,parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi
  
def get_ffme_returns(col_1='Lo 10',col_2='Hi 10', name_1='Small Cap', name_2='Large Cap'):
    """
    Load the Fama-French Market Equity dataset for the Top and Bottom Deciles by market cap
    """
    me_monthly = pd.read_csv('./data/Portfolios_Formed_on_ME_monthly_EW.csv',
                            header=0,index_col=0,na_values=-99.99)
   
    returns = me_monthly[[col_1, col_2]]
    returns.columns = [name_1, name_2]
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format='%Y%m').to_period('M')
    return returns
  
def get_ind30_vw_returns():
  """
  Load the industry 30 monthly returns
  """
  ind = pd.read_csv('./data/ind30_m_vw_rets.csv',header=0, index_col=0, parse_dates=True)/100
  ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
  ind.columns = ind.columns.str.strip()
  return ind

def get_ind30_size():
  """
  Load the industry 30 monthly returns
  """
  ind = pd.read_csv('./data/ind30_m_size.csv',header=0, index_col=0, parse_dates=True)
  ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
  ind.columns = ind.columns.str.strip()
  return ind

def get_ind30_nfirms():
  """
  Load the industry 30 monthly returns
  """
  ind = pd.read_csv('./data/ind30_m_nfirms.csv',header=0, index_col=0, parse_dates=True)
  ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
  ind.columns = ind.columns.str.strip()
  return ind