import os

import matplotlib

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf as _plot_acf
from statsmodels.graphics.tsaplots import plot_pacf as _plot_pacf
from statsmodels.graphics.tsaplots import quarter_plot
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils import data_utils

sns.set()


def plot(series, label=''):
    series.plot()
    # plt.title(label + ' Plot')
    plt.title('')
    plt.xlabel('Time')
    plt.ylabel(label)


def plot_acf(series, label=''):
    _plot_acf(series)
    # plt.title(label + ' ACF Plot')
    plt.title('')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')


def plot_pacf(series, label=''):
    _plot_pacf(series)
    # plt.title(label + ' PACF Plot')
    plt.title('')
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')


def plot_dist(series, label=''):
    sns.distplot(series)
    # plt.title(label + ' Distribution Plot')
    plt.title('')
    plt.xlabel('Value')
    plt.ylabel('Frequency')


def plot_quarter(series, label=''):
    quarter_plot(series)
    # plt.title(label + ' Seasonal Subseries Plot')
    plt.title('')
    plt.xlabel('Quarter')
    plt.ylabel(label)


def plot_lag(series, label=''):
    lag_plot(series)
    # plt.title(label + ' Lag Plot')
    plt.title('')
    plt.xlabel(label + ' (t)')
    plt.ylabel(label + ' (t+1)')


def stationary_test(series, label=''):
    df = pd.DataFrame(index=pd.Index(('ADF', 'KPSS'), name='test'),
                      columns=['statistic', 'p-value'])
    df.at['ADF', :] = adfuller(series)[:2]
    df.at['KPSS', :] = kpss(series)[:2]

    return df


class SeriesPlotter:

    def __init__(self, series, label, output=None):
        self.series = series
        self.label = ' '.join(label.split('_'))
        self.file_name = '_'.join(label.split(' '))
        self.output = output
        if output is not None and not os.path.exists(output):
            os.mkdir(output)

    def plot(self):
        plot(self.series, self.label)
        if not self.output:
            plt.show()
        else:
            plot_path = os.path.join(self.output, self.file_name + '_plot.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    def plot_acf(self):
        plot_acf(self.series, self.label)
        if not self.output:
            plt.show()
        else:
            plot_path = os.path.join(self.output, self.file_name + '_acf.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    def plot_pacf(self):
        plot_pacf(self.series, self.label)
        if not self.output:
            plt.show()
        else:
            plot_path = os.path.join(self.output, self.file_name + '_pacf.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    def plot_dist(self):
        plot_dist(self.series, self.label)
        if not self.output:
            plt.show()
        else:
            plot_path = os.path.join(self.output, self.file_name + '_dist.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    def plot_quarter(self):
        plot_quarter(self.series, self.label)
        if not self.output:
            plt.show()
        else:
            plot_path = os.path.join(self.output, self.file_name + '_quarter.pdf')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    def stationary_test(self):
        test = stationary_test(self.series, self.label)
        if not self.output:
            print test
        else:
            plot_path = os.path.join(self.output, self.file_name + '_test.csv')
            test.to_csv(plot_path)

        return

    def plot_all(self):
        self.plot()
        self.plot_acf()
        self.plot_pacf()
        self.plot_dist()
        self.plot_quarter()
        self.stationary_test()


class DataPlotter:

    def __init__(self, data_frame, country, output=None):
        self.output = output
        if output is not None and not os.path.exists(output):
            os.mkdir(output)

        self.data = data_frame
        self.country = country
        self.columns = data_frame.columns.tolist()
        self.labels = [' '.join([country, col]) for col in self.columns]
        self.plotters = {}

        for column, label in zip(self.columns, self.labels):
            self.plotters[column] = SeriesPlotter(self.data[column], label, output=output)

    def plot_matrix(self, labels=None):
        sns.pairplot(self.data)
        if not self.output:
            plt.show()
        else:
            plot_file = os.path.join(self.output, self.country + '_pairplot.pdf')
            plt.savefig(plot_file)
            plt.close()

    def plot(self, labels=None):
        if labels is None:
            labels = self.columns

        for label in labels:
            self.plotters[label].plot()

    def plot_acf(self, labels):
        labels = self.columns if labels is None else labels

        for label in labels:
            self.plotters[label].plot_acf()

    def plot_pacf(self, labels=None):
        if labels is None:
            labels = self.columns

        for label in labels:
            self.plotters[label].plot_pacf()

    def plot_dist(self, labels=None):
        if labels is None:
            labels = self.columns

        for label in labels:
            self.plotters[label].plot_dist()

    def plot_quarter(self, labels=None):
        if labels is None:
            labels = self.columns

        for label in labels:
            self.plotters[label].plot_quarter()

    def stationary_test(self, labels=None):
        if labels is None:
            labels = self.columns

        for label in labels:
            self.plotters[label].stationary_test()

    def plot_all(self, labels=None):
        if labels is None:
            labels = self.columns

        for label in labels:
            self.plotters[label].plot_all()

        self.plot_matrix()


def stationarity_tests(output):
    data = data_utils.get_data_dict()

    index = ['_'.join([i, j]) for i in data_utils.COUNTRIES for j in data_utils.VARIABLES]
    columns = ['ADF stat', 'ADF p-val', 'KPSS stat', 'KPSS p-val']
    df = pd.DataFrame(index=index, columns=columns)
    for country in data_utils.COUNTRIES:
        for var in data_utils.VARIABLES:
            index = '_'.join([country, var])

            adf_res = adfuller(data[country][var])[:2]
            kpss_res = kpss(data[country][var])[:2]

            df.loc[index, :] = adf_res + kpss_res

    df.to_csv(os.path.join(output, 'stationary_tests.csv'))


# TODO Implement cross correlation matrix for each variable with different lags of other variables

if __name__ == '__main__':
    output = 'data-analysis'

    ea = data_utils.get_ea_data()
    plotter = DataPlotter(ea, 'EA', output)
    plotter.plot_all()

    us = data_utils.get_us_data()
    plotter = DataPlotter(us, 'US', output)
    plotter.plot_all()

    stationarity_tests(output)
