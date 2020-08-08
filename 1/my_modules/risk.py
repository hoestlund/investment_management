import pandas as pd

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
