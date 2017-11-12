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

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'experiments', 'python-analysis')

def plot_rolling_stats(data, window=12):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    for index, variable in enumerate(data.columns):
        plt.figure()
        data[variable].plot(color='blue', label='Original')
        rolling_mean[variable].plot(color='red', label='Rolling Mean')
        rolling_std[variable].plot(color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title(variable)


def stationary_tests(data, country):
    results = pd.DataFrame(index=['ADF', 'PP', 'KPSS'],
                           columns=['t-stat', 'p-value', 'crit-val 1%', 'crit-val 5%', 'crit-val 10%', 'result'])
    for variable in data.columns:
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

        file_path = os.path.join(OUTPUT_PATH, 'stationary-tests', variable + '_' + country + '.csv')
        results.to_csv(file_path,float_format='%.3f')


def main():
    ea_data = data_utils.get_ea_data()
    us_data = data_utils.get_us_data()

    stationary_tests(ea_data, 'EA')
    stationary_tests(us_data, 'US')

    plt.show()


if __name__ == '__main__':
    main()
