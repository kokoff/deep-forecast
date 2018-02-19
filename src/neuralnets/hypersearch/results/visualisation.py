import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
import itertools
from src.utils import data_utils
import numpy as np



sns.set()


def plot_predictions(df, country, var, path):
    df.plot()
    plt.ylabel(country + ' ' + var)
    plt.savefig(path)


class ResultsPlotter:
    def __init__(self, path, output_dir, show=False):
        self.show = False
        self.path = path
        self.out = output_dir
        if isinstance(self.path, pd.DataFrame):
            self.data = self.path
        else:
            self.data = pd.read_csv(path, index_col=False)

    def _remove_cols(self):
        cols_to_drop = [i for i in self.data.columns if len(self.data[i].drop_duplicates().dropna()) <= 1]
        return self.data.drop(cols_to_drop, axis=1)

    def pairplot(self):
        sns.set()
        data = self._remove_cols()
        data.sort_values('val', ascending=True, inplace=True)
        data = pd.get_dummies(data)

        if len(data) > 6:
            data['val'], bins = pd.qcut(data['val'], [0, .05, .1, .25, .5, 1.], retbins=True)

        # data = data.sort_values('val', ascending=False).reset_index(drop=True)
        data.sort_values('val', ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)
        cols = [i for i in data.columns if i not in ['train', 'val']]

        g = sns.pairplot(data, hue='val', vars=cols,
                         palette=sns.color_palette('hls'))

        path = os.path.join(self.out, 'pairplot.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def pairplot_val(self):
        sns.set()
        data = self._remove_cols()
        data.sort_values('val', ascending=True, inplace=True)
        data = pd.get_dummies(data)

        cols = [i for i in data.columns if i not in ['train', 'val']]

        g = sns.pairplot(data, x_vars=cols, y_vars=['train', 'val'], kind='scatter')
        g.set(yscale='log')
        g.map(plt.scatter)

        path = os.path.join(self.out, 'pairplot_val.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def parallel_coordinates(self):
        data = self._remove_cols()
        data = data.sort_values('val', ascending=False)
        data = pd.get_dummies(data)

        if len(data) > 6:
            data['val'], bins = pd.qcut(data['val'], [0, .05, .1, .25, .5, 1.], retbins=True)

        cols = [i for i in data.columns if i not in ['train', 'val']]
        ticks = np.unique(data[cols].values.flatten())
        ticks = np.delete(ticks, ticks[ticks == 0])

        len_bins = len(data) if len(data) < 6 else len(bins)
        colors = sns.color_palette('hls', len_bins)
        ax = parallel_coordinates(data, 'val', cols=cols, color=colors)
        ax.set_yscale('log')
        ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().get_major_formatter().labelOnlyBase = False

        path = os.path.join(self.out, 'parallel_coordinates.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def cond_one_to_many(self):
        data = self._remove_cols()
        data = data.sort_values('val', ascending=False)
        # data = pd.get_dummies(data)

        cols = list(data.columns[:-2])
        cols.sort(key=lambda x: len(data[x].dropna().unique()), reverse=True)
        if len(cols) < 2:
            return

        param = cols.pop(0)

        new_data = pd.DataFrame(columns=['params', 'val', 'train'], index=data.index)
        for i in range(len(data)):
            new_data.at[i, 'params'] = '(' + ','.join([str(data.at[i, j]) for j in cols]) + ')'

        new_data[param] = data[param]
        new_data['val'] = data['val']
        new_data['train'] = data['train']

        sns.set()
        col_wrap = 4 if len(new_data['params'].unique()) >= 4 else len(new_data['params'].unique())
        g = sns.FacetGrid(new_data, col='params', col_wrap=col_wrap)
        g = g.map(plt.plot, param, 'val', color='r', label='val')
        g = g.map(plt.plot, param, 'train', color='b', label='train')
        g.add_legend()

        path = os.path.join(self.out, 'conditional_many.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def cond_one_to_one(self):
        cols = list(self.data.columns[:-2])
        cols.sort(key=lambda x: len(self.data[x].dropna().unique()), reverse=True)
        cols = cols[:2] + [i for i in cols[2:] if len(self.data[i].dropna().unique()) > 1]

        combs = list(itertools.combinations(cols, 2))

        for i, comb in enumerate(combs):
            sns.set()
            col_wrap = 4 if len(self.data[comb[1]].unique()) >= 4 else len(self.data[comb[1]].unique())
            g = sns.FacetGrid(self.data, col=comb[1], col_wrap=col_wrap)
            g = g.map(plt.plot, comb[0], 'train', color='b', label='train')
            g = g.map(plt.plot, comb[0], 'val', color='r', label='val')
            g.add_legend()

            path = os.path.join(self.out, 'conditional' + str(i) + '.pdf')
            plt.savefig(path)
            if self.show:
                plt.show()

    def plot_all(self):
        try:
            self.pairplot()
            plt.close()
        except Exception as e:
            print e
        try:
            self.pairplot_val()
            plt.close()
        except Exception as e:
            print e
        try:
            self.parallel_coordinates()
            plt.close()
        except Exception as e:
            print e
        try:
            self.cond_one_to_many()
            plt.close()
        except Exception as e:
            print e
        try:
            self.cond_one_to_one()
            plt.close()
        except Exception as e:
            print e


def main():
    log = pd.read_csv('/home/skokov/project/src/neuralnets/temp/EA_CPI/log.csv')

    pltr = ResultsPlotter('/home/skokov/project/src/neuralnets/temp/EA_CPI/log.csv', '../temp', show=True)
    pltr.cond_one_to_many()


if __name__ == '__main__':
    main()
