import numpy as np
import pandas as pd


def test():
  return 'hello world'
  
def getJul14Data():
  uber_data = pd.read_csv('data/uber/uber-raw-data-jul14.csv')
  uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'])
  uber_data['Date/Time'].dt.floor('1H').head(10)
  return uber_data