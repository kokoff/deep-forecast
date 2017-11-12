import os

import pandas as pd
from arch.unitroot import ADF, PhillipsPerron, KPSS
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from code.utils import data_utils
from code.utils.experiments_utils import expr_sub_dir, expr_file_name

from statsmodels.tsa.seasonal import seasonal_decompose

EXPERIMENTS_DIR_NAME = 'python-analysis'
ROLLING_STATISTICS_DIR = expr_sub_dir(EXPERIMENTS_DIR_NAME, 'rolling-statistics')
STATIONARY_TESTS_DIR = expr_sub_dir(EXPERIMENTS_DIR_NAME, 'stationary-tests')
ACF_PACF_DIR = expr_sub_dir(EXPERIMENTS_DIR_NAME, 'acf-pacf')
DECOMPOSITION_DIR = expr_sub_dir(EXPERIMENTS_DIR_NAME, 'decomposition')


def plot_rolling_stats(series, window=12, country=None, variable=None, out=False):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    plt.figure()
    series.plot(color='blue', label='Original')
    rolling_mean.plot(color='red', label='Rolling Mean')
    rolling_std.plot(color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(variable)

    if country and variable and out:
        file_path = os.path.join(ROLLING_STATISTICS_DIR, expr_file_name(country, variable, 'png'))
        plt.savefig(file_path, bbox_inches='tight')


def stationary_tests(series, country=None, variable=None, out=False):
    results = pd.DataFrame(index=['ADF', 'PP', 'KPSS'],
                           columns=['t-stat', 'p-value', 'crit-val 1%', 'crit-val 5%', 'crit-val 10%', 'result'])

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

    print '\n', variable, '\n', results, '\n'

    if country and variable and out:
        file_path = os.path.join(STATIONARY_TESTS_DIR, expr_file_name(country, variable, 'csv'))
        results.to_csv(file_path, float_format='%.3f')


def acf_and_pacf(series, country=None, variable=None, out=False):
    plt.figure()

    ax = plt.subplot(1, 2, 1)
    plot_acf(series, ax=ax)

    ax = plt.subplot(1, 2, 2)
    plot_pacf(series, ax=ax)

    if country and variable and out:
        file_path = os.path.join(ACF_PACF_DIR, expr_file_name(country, variable, 'png'))
        plt.savefig(file_path, bbox_inches='tight')


def decompose_data(series, country=None, variable=None, out=False):
    series.index = series.index.to_timestamp()
    result = seasonal_decompose(series)
    result.resid.trend = result.trend.index.to_period()
    result.seasonal.index = result.seasonal.index.to_period()
    result.observed.index = result.observed.index.to_period()
    result.resid.index = result.resid.index.to_period()
    result.plot()

    if country and variable and out:
        file_path = os.path.join(DECOMPOSITION_DIR, expr_file_name(country, variable, 'png'))
        plt.savefig(file_path, bbox_inches='tight')

    return result.resid.dropna()


def main():
    data = data_utils.get_data()
    output = True

    for country, data_frame in data.items():
        for variable in data_frame.columns:
            series = data_frame[variable].dropna()

            stationary_tests(series, country=country, variable=variable, out=output)
            plot_rolling_stats(series, country=country, variable=variable, out=output)
            acf_and_pacf(series, country=country, variable=variable, out=output)

            decomposed_series = decompose_data(series, country=country, variable=variable, out=output)
            stationary_tests(decomposed_series, country=country, variable='decomposed_' + variable, out=output)
            plot_rolling_stats(decomposed_series, country=country, variable='decomposed_' + variable, out=output)
            acf_and_pacf(decomposed_series, country=country, variable='decomposed_' + variable, out=output)

            differenced_series = series.diff().dropna()
            stationary_tests(differenced_series, country=country, variable='differenced_' + variable, out=output)
            plot_rolling_stats(differenced_series, country=country, variable='differenced_' + variable, out=output)
            acf_and_pacf(differenced_series, country=country, variable='differenced_' + variable, out=output)

    # plt.show()


if __name__ == '__main__':
    main()
