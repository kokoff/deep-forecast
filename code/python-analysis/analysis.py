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


def plot_rolling_stats(data, window=12, country=None):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    for index, variable in enumerate(data.columns):
        plt.figure()
        data[variable].plot(color='blue', label='Original')
        rolling_mean[variable].plot(color='red', label='Rolling Mean')
        rolling_std[variable].plot(color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title(variable)

        if country:
            file_path = os.path.join(ROLLING_STATISTICS_DIR, expr_file_name(country, variable, 'png'))
            plt.savefig(file_path, bbox_inches='tight')


def stationary_tests(data, country=None):
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

        print '\n', variable, '\n', results, '\n'

        if country:
            file_path = os.path.join(STATIONARY_TESTS_DIR, expr_file_name(country, variable, 'csv'))
            results.to_csv(file_path, float_format='%.3f')


def acf_and_pacf(data, country=None):
    for variable in data.columns:
        series = data[variable].dropna()

        plt.figure()

        ax = plt.subplot(1,2,1)
        plot_acf(series,ax=ax)

        ax = plt.subplot(1, 2, 2)
        plot_pacf(series, ax=ax)

        if country:
            file_path = os.path.join(ACF_PACF_DIR, expr_file_name(country, variable, 'png'))
            plt.savefig(file_path, bbox_inches='tight')




def main():
    ea_data = data_utils.get_ea_data()
    us_data = data_utils.get_us_data()
    data = data_utils.get_data()

    for country, df in data.items():
        stationary_tests(ea_data, country=country)
        plot_rolling_stats(ea_data, country=country)
        acf_and_pacf(ea_data, country=country)

    #plt.show()


if __name__ == '__main__':
    main()
