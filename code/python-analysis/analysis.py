import os
import numpy as np
import pandas as pd
from pandas.plotting import table
from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.unitroot import ADF, PhillipsPerron, KPSS
from code import data_utils

#########################################################################
# Read Data
#########################################################################
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, '..')
ea_data = os.path.join(data_dir, 'EA.csv')
us_data = os.path.join(data_dir, 'US.csv')

print 'EA Data: ', ea_data
print 'US Data: ', us_data

ea_data = data_utils.get_ea_data()
us_data = data_utils.get_us_data()

# def test_stationarity(timeseries):
#     adftest = adfuller(timeseries, autolag='t-stat')
#     kpsstest = kpss(timeseries)
#
#     tests = DataFrame(np.nan, index=[i for i in range(8)], columns=[i for i in range(2)])
#     print tests
#
#     tests[0][0:4] = list(adftest[0:4])
#     tests[1][0:3] = list(kpsstest[0:3])
#
#     tests.columns = ['ADF', 'KPSS']
#     tests.index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used', 'Critical Value (10%)',
#                    'Critical Value (5%)', 'Critical Value (2.5%)', 'Critical Value (1%)']
#
#     for key, value in adftest[4].items():
#         tests['ADF']['Critical Value (%s)' % key] = value
#
#     for key, value in kpsstest[3].items():
#         tests['KPSS']['Critical Value (%s)' % key] = value
#
#     print tests


def plot_rolling_stats(data, window=12):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    for index, variable in enumerate(data.columns):
        print pd.PeriodIndex(data.index, freq='Q')
        plt.figure()
        data[variable].plot(color='blue', label='Original')
        rolling_mean[variable].plot(color='red', label='Rolling Mean')
        rolling_std[variable].plot(color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title(variable)


def stationary_tests(data):
    results = pd.DataFrame(index=['ADF', 'PP', 'KPSS'],
                           columns=['t-stat', 'p-value', 'crit-val 1%', 'crit-val 5%', 'crit-val 10%', 'result'])
    for variable in data.columns:
        print '----------------------------------------'
        print variable
        print '----------------------------------------'
        series = data[variable].dropna()
        adf = ADF(series)

        pp = PhillipsPerron(series)

        kpss = KPSS(series)

        for test, test_name in zip([adf, pp, kpss], ['ADF', 'PP', 'KPSS']):
            results['t-stat'][test_name] = test.stat
            results['p-value'][test_name] = test.pvalue
            results['crit-val 1%'][test_name] = test.critical_values['1%']
            results['crit-val 5%'][test_name] = test.critical_values['5%']
            results['crit-val 10%'][test_name] = test.critical_values['10%']
            results['result'][test_name] = test.alternative_hypothesis if test.pvalue < 0.05 else test.null_hypothesis

        print results


stationary_tests(ea_data)

# test_stationarity(ea_data['CPI'].dropna(), 8)
#
# # plot_roll(ea_data, 12)
# decomp = seasonal_decompose(ea_data['CPI'])
# decomp.plot()
# plt.show()
# plt.plot(decomp.resid + decomp.seasonal + decomp.trend)
# plt.plot(ea_data['CPI'], '--r')
# plt.show()
# test_stationarity(decomp.resid.dropna(), 16)
plt.show()
