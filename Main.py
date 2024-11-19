import pandas as pd

data = pd.read_csv('temperatures.csv', sep=',', header=0, index_col=0, decimal=',')
n = len(data)
data = data.drop
