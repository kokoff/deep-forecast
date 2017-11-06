import os
import numpy as np
import pandas as pd
from pandas.plotting import table
from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose


#########################################################################
# Read Data
#########################################################################
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, '..')
ea_data = os.path.join(data_dir, 'ea.csv')
us_data = os.path.join(data_dir, 'us.csv')

print 'EA Data: ', ea_data
print 'US Data: ', us_data

ea_data = DataFrame.from_csv(ea_data)
us_data = DataFrame.from_csv(us_data)


def test_stationarity(timeseries, window=12):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    # rolmean = expwighted_avg = pd.ewma(timeseries, halflife=window)
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics:
    timeseries.plot(color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')
    rolstd.plot(color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    adftest = adfuller(timeseries,autolag='t-stat')
    kpsstest = kpss(timeseries)

    tests = DataFrame(np.nan, index=[i for i in range(8)], columns=[i for i in range(2)])
    print tests

    tests[0][0:4] = list(adftest[0:4])
    tests[1][0:3] = list(kpsstest[0:3])

    tests.columns = ['ADF', 'KPSS']
    tests.index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used', 'Critical Value (10%)',
                   'Critical Value (5%)', 'Critical Value (2.5%)', 'Critical Value (1%)']

    for key, value in adftest[4].items():
        tests['ADF']['Critical Value (%s)' % key] = value

    for key, value in kpsstest[3].items():
        tests['KPSS']['Critical Value (%s)' % key] = value

    print tests

def plot_roll(data_frame, window=4):
    rollmean = pd.ewma(data_frame, halflife=1)
    print rollmean
    rollmean.columns = [i + ' Roll Avg ' + str(window) for i in rollmean.columns]
    rollstd = data_frame.rolling(window=4).std()
    rollstd.columns = [i + ' Roll Std ' + str(window) for i in rollstd.columns]

    ax = rollmean.plot(subplots=True, layout=(4, 2), sharex=False, style='r-')
    data_frame.plot(subplots=True, layout=(4, 2), sharex=False, ax=ax.flatten()[:-1], style='b-')
    rollstd.plot(subplots=True, layout=(4, 2), sharex=False, ax=ax.flatten()[:-1], style='k-')


# test_stationarity(ea_data['UR'].dropna(), 8)

# plot_roll(ea_data, 12)
decomp = seasonal_decompose(ea_data['CPI'])
decomp.plot()
plt.show()
plt.plot(decomp.resid + decomp.seasonal + decomp.trend)
plt.plot(ea_data['CPI'], '--r')
plt.show()
test_stationarity(decomp.resid.dropna(), 16)
plt.show()


