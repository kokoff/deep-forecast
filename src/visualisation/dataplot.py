import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, quarter_plot
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils import data_utils


class VariablePlotter:

    def __init__(self, country, variable, directory='temp', output=False, show=True):
        self.output = output
        self.show = show
        self.dir = directory
        self.country = country
        if isinstance(variable, pd.Series):
            self.data = variable
            self.variable = variable.name
        else:
            self.data = data_utils.get_data_frame()[(country, variable)]
            self.variable = variable
        self.label = self.country + ' ' + self.variable

    def path(self, name):
        basename = '_'.join([self.country, self.variable, name])
        return os.path.join(self.dir, basename)

    def plot(self):
        self.data.plot()
        plt.title(self.label + ' Plot')
        plt.xlabel('Time')
        plt.ylabel(self.label)

        if self.output:
            plt.savefig(self.path('plot.pdf'))
        plt.show()

    def plot_acf(self):
        plot_acf(self.data)
        plt.title(self.label + ' ACF Plot')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')

        if self.output:
            plt.savefig(self.path('acf.pdf'))
        if self.show:
            plt.show()

    def plot_pacf(self):
        plot_pacf(self.data)
        plt.title(self.label + ' PACF Plot')
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')

        if self.output:
            plt.savefig(self.path('pacf.pdf'))
        if self.show:
            plt.show()

    def plot_dist(self):
        sns.distplot(self.data)
        plt.title(self.label + ' Distribution Plot')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        if self.output:
            plt.savefig(self.path('dist.pdf'))
        if self.show:
            plt.show()

    def plot_quarter(self):
        quarter_plot(self.data)
        plt.title(self.label + ' Seasonal Subseries Plot')
        plt.xlabel('Quarter')
        plt.ylabel(self.label)

        if self.output:
            plt.savefig(self.path('seasonal.pdf'))
        if self.show:
            plt.show()

    def stationary_test(self):
        df = pd.DataFrame(index=pd.Index(('ADF', 'KPSS'), name='test'),
                          columns=['statistic', 'p-value'])
        df.at['ADF', :] = adfuller(self.data)[:2]
        df.at['KPSS', :] = kpss(self.data)[:2]

        if self.output:
            df.to_csv(self.path('tests.csv'))
        if self.show:
            print df
        return df

    def plot_all(self):
        self.plot()
        self.plot_acf()
        self.plot_pacf()
        self.plot_dist()
        self.plot_quarter()
        self.stationary_test()


class DataPlotter:

    def __init__(self, directory='temp', output=False, show=True):
        self.output = output
        self.show = show
        self.directory = directory
        self.data = data_utils.get_data_frame()
        self.plotters = {}
        for tup in self.data.columns.tolist():
            self.plotters[tup] = VariablePlotter(tup[0], tup[1], directory=directory, output=output, show=show)

    def plot_matrix(self):
        data = data_utils.get_flat_data_frame()
        x_vars = data.columns[len(data.columns) / 2:]
        y_vars = data.columns[:len(data.columns) / 2]
        sns.pairplot(data, x_vars=x_vars, y_vars=y_vars)
        if self.output:
            plt.savefig(os.path.join(self.directory, 'matrix.pdf'))
        if self.show:
            plt.show()

        for country in self.data.columns.levels[0]:
            sns.pairplot(self.data[country])
            if self.output:
                plt.savefig(os.path.join(self.directory, country + '_matrix.pdf'))
            if self.show:
                plt.show()

    def _get_params(self, countries=None, variables=None):
        if countries is None:
            countries = list(self.data.columns.levels[0])
        elif not isinstance(countries, list):
            countries = [countries]

        if variables is None:
            variables = list(self.data.columns.levels[1])
        elif not isinstance(variables, list):
            variables = [variables]

        params = []
        for i in countries:
            for j in variables:
                params.append((i, j))

        return params

    def plot(self, countries=None, variables=None):
        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].plot()

    def plot_acf(self, countries=None, variables=None):
        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].plot_acf()

    def plot_pacf(self, countries=None, variables=None):
        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].plot_pacf()

    def plot_dist(self, countries=None, variables=None):
        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].plot_dist()

    def plot_quarter(self, countries=None, variables=None):
        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].plot_quarter()

    def stationary_test(self, countries=None, variables=None):
        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].stationary_test()

    def plot_all(self, countries=None, variables=None):
        self.plot_matrix()

        params = self._get_params(countries, variables)
        for param in params:
            self.plotters[param].plot_all()


# TODO Implement cross correlation matrix for each variable with different lags of other variables

if __name__ == '__main__':
    plotter = DataPlotter(output=True)
    plotter.plot_all()
