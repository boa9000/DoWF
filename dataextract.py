from netCDF4 import Dataset
import pandas as pd
import numpy as np
from floris.tools import FlorisInterface, WindRose
import matplotlib.pyplot as plt

hubheight = 120
windshear = 0.17

nc = Dataset('Kast.nc')


df = pd.DataFrame()
df['ws'] = nc.variables['WS10'][:] / ((10/ hubheight)**windshear)
df['wd'] = nc.variables['WD10'][:]
df = df.set_index(nc.variables['time'][:])


df['wd'] = (df['wd']/150).round(decimals=1) * 150

df['ws'] = (df['ws']/1).round(decimals=1) * 1

df['wd'].replace(360, 0, inplace=True)


dff = df.groupby(['ws', 'wd']).value_counts().reset_index()
dff['freq_val'] = dff['count']/dff['count'].sum()
dff.drop('count', axis = 1, inplace=True)

print(dff)

dff.to_csv('windrose_den_test9090.csv', index=False)


wr = WindRose()

rose = wr.read_wind_rose_csv(filename='windrose_den_test9090.csv')

wr.plot_wind_rose()

plt.show()
