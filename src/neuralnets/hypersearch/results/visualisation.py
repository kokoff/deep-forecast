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
from matplotlib import ticker

sns.set()


def plot_predictions(df, country, var, path):
    df.plot()
    plt.ylabel(country + ' ' + var)
    plt.savefig(path)
    plt.close('all')


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
        data = data.fillna(0)

        if len(data) > 6:
            data['val'], bins = pd.qcut(data['val'], [0, .05, .1, .25, .5, 1.], retbins=True)

        # data = data.sort_values('val', ascending=False).reset_index(drop=True)
        data.sort_values('val', ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)
        cols = [i for i in data.columns if i not in ['train', 'val']]

        g = sns.pairplot(data, hue='val', vars=cols,
                         palette=sns.color_palette('hls'))

        path = os.path.join(self.out, 'pairplot.pdf')
        plt.savefig(path, bbox_inches='tight')
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
        plt.ylabel('val')

        path = os.path.join(self.out, 'pairplot_val.pdf')
        plt.savefig(path, bbox_inches='tight')
        if self.show:
            plt.show()

    def parallel_coordinates(self):
        data = self._remove_cols()
        data = data.sort_values('val', ascending=False)
        data = pd.get_dummies(data)

        parallel_coords1(data)

        path = os.path.join(self.out, 'parallel_coordinates.pdf')
        plt.savefig(path, bbox_inches='tight')
        if self.show:
            plt.show()

    def plot_all(self):
        self.pairplot()
        plt.close('all')

        self.pairplot_val()
        plt.close()
        plt.close('all')

        self.parallel_coordinates()
        plt.close('all')
        # try:
        #     self.cond_one_to_many()
        #     plt.close()
        # except Exception as e:
        #     print e
        # plt.close('all')
        # try:
        #     self.cond_one_to_one()
        #     plt.close()
        # except Exception as e:
        #     print e
        # plt.close('all')


def parallel_coords1(df):
    cols = df.columns[:-2]

    if len(df) > 6:
        df['val'], bins = pd.qcut(df['val'], [0, .05, .1, .25, .5, 1.], retbins=True)
    else:
        bins = np.squeeze(df['val'].values)

    x = [i for i, _ in enumerate(cols)]
    colours = sns.hls_palette(len(bins))

    # create dict of categories: colours
    # colours = {df.loc[i, 'val']: colours[i] for i in df.index}

    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(15, 5))

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            mpg_category = df.loc[idx, 'val']
            ax.plot(x, df.loc[idx, cols], colours)
        ax.set_xlim([x[i], x[i + 1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks - 1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks - 1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]])

    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    # Add legend to plot
    plt.legend(
        [plt.Line2D((0, 1), (0, 0), color=colours[idx]) for idx in range(len(bins))],
        df['val'].unique(),
        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

    # plt.show()


def main():
    # print sns.hls_palette(10)

    log = pd.read_csv('/home/skokov/project/src/neuralnets/models/mlp_experiments/EA_[one]_[one]4/EA_CPI/log.csv')

    pltr = ResultsPlotter('/home/skokov/mlp_experiments/EA_[one]_[one]_8/EA_LR10/log.csv',
                          '/home/skokov/mlp_experiments/EA_[one]_[one]_8/EA_LR10/parameter_figures',
                          show=True)
    pltr.plot_all()


if __name__ == '__main__':
    main()
