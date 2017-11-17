import sys, os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')))

import pandas as pd
from arch.unitroot import ADF, PhillipsPerron, KPSS
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from src.utils import data_utils as du


def plot_rolling_stats(series, window=8):
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()

    plt.figure()
    series.plot(color='blue')
    rolling_mean.plot(color='red', label=str(series.name) + ' rolling mean')
    rolling_std.plot(color='black', label=str(series.name) + ' rolling std')
    plt.legend()


def stationary_tests(series):
    du.remove_na(series)

    results = pd.DataFrame(index=['ADF', 'PP', 'KPSS'],
                           columns=['t-stat', 'p-value', 'crit-val 1%', 'crit-val 5%', 'crit-val 10%', 'result'])

    adf = ADF(series)
    pp = PhillipsPerron(series)
    kpss = KPSS(series)

    for test, test_name in zip([adf, pp, kpss], ['ADF', 'PP', 'KPSS']):
        results.at[test_name, 't-stat'] = test.stat
        results.at[test_name, 'p-value'] = test.pvalue
        results.at[test_name, 'crit-val 1%'] = test.critical_values['1%']
        results.at[test_name, 'crit-val 5%'] = test.critical_values['5%']
        results.at[test_name, 'crit-val 10%'] = test.critical_values['10%']
        results.at[test_name, 'result'] = test.alternative_hypothesis if test.pvalue < 0.05 else test.null_hypothesis
        results[results.columns[:-1]] = results[results.columns[:-1]].apply(pd.to_numeric)

    return results


def plot_auto_correlation(series):
    du.remove_na(series)
    plt.figure()

    ax = plt.subplot(1, 2, 1)
    plot_acf(series, ax=ax)

    ax = plt.subplot(1, 2, 2)
    plot_pacf(series, ax=ax)
    plt.legend()


def plot_series(series):
    plt.figure()
    series.plot()
    plt.legend()


def main():
    data = du.get_us_data()
    series = data['UR']
    plot_series(series)

    plt.show()


if __name__ == '__main__':
    main()
