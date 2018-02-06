import itertools
import os

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
from keras import backend as K
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import ParameterGrid

from src.neuralnets.forecast_models import ModelWrapper
from src.utils.data_utils import get_data_formatted


class GridSearch:

    def __init__(self, build_fn, param_dict, directory='temp'):
        self.dir = directory
        self.estimator = ModelWrapper(build_fn)
        self.param_dict = param_dict
        self.num_runs = 20

        self.run_results = None
        self.parameter_results = None
        self.best_estimator = ModelWrapper(build_fn)
        self.best_performance = None

    def set_dir(self, directory):
        self.dir = directory

    def reset_logs(self):
        columns = ['train', 'val']
        self.run_results = pd.DataFrame(columns=columns)

        columns = self.param_dict.keys() + ['train', 'val']
        self.parameter_results = pd.DataFrame(columns=columns)

        self.best_performance = np.inf

    def grid_search(self, x_train, y_train, x_val, y_val):
        self.param_dict['input_dim'] = [x_train.shape[1]]
        self.param_dict['output_dim'] = [y_train.shape[1]]
        self.reset_logs()

        param_grid = ParameterGrid(self.param_dict)

        for i, params in enumerate(param_grid):
            K.clear_session()

            for j in range(self.num_runs):
                print 'Par', i, 'Run ', j, '\tParameters ', str(params)
                self.estimator.set_params(**params)

                self.estimator.fit(x_train, y_train, validation_data=[x_val, y_val])
                self.evaluate_run(x_train, y_train, x_val, y_val, j)

            self.evaluate_parameters(params, i)

        self.best_estimator.fit(x_train, y_train)
        self.save_results()
        return self.best_estimator.model

    def grid_search_data(self, data_dict):
        data_params = ParameterGrid(data_dict)
        for k, data_param in enumerate(data_params):
            print data_param
            self.dir = data_param['country'] + '_' + data_param['var_dict']['y'][0]

            x_train, y_train, x_val, y_val, x_test, y_test = get_data_formatted(**data_param)
            print 'Data', k, data_param

            model = self.grid_search(x_train, y_train, x_val, y_val)
            self.test_final_model(model, x_val, y_val, x_test, y_test)

    def evaluate_run(self, x_train, y_train, x_val, y_val, run_num):
        self.run_results.at[run_num, 'train'] = self.estimator.score(x_train, y_train)
        self.run_results.at[run_num, 'val'] = self.estimator.score(x_val, y_val)

    def evaluate_parameters(self, params, index):
        self.parameter_results.at[index, params.keys()] = params.values()
        self.parameter_results.at[index, ['train', 'val']] = self.run_results.mean().values
        self.evaluate_model(self.parameter_results.at[index, 'val'], params)

    def evaluate_model(self, performance, params):
        if self.best_performance is None or self.best_performance > performance:
            self.best_performance = performance
            self.best_estimator.set_params(**params)

    def test_final_model(self, model, x_val, y_val, x_test, y_test):
        best_results = pd.DataFrame(columns=['val prediction', 'val forecast', 'test prediction', 'test forecast'])
        best_results.at[0, 'val prediction'] = model.evaluate(x_val, y_val)
        best_results.at[0, 'val forecast'] = model.evaluate_forecast(x_val, y_val)
        best_results.at[0, 'test prediction'] = model.evaluate(x_test, y_test)
        best_results.at[0, 'test forecast'] = model.evaluate_forecast(x_test, y_test)

        path = os.path.join('gridsearch', self.dir, 'best_results.csv')
        best_results.to_csv(path)

        val_real = pd.DataFrame(y_val.values, index=y_val.index, columns=['real val'])
        test_real = pd.DataFrame(y_test.values, index=y_test.index, columns=['real test'])
        val_prediction = pd.DataFrame(model.predict(x_val), index=y_val.index, columns=['prediction val'])
        test_prediction = pd.DataFrame(model.predict(x_test), index=y_test.index, columns=['prediction test'])
        val_forecast = pd.DataFrame(model.forecast(x_val, y_val), index=y_val.index[1:], columns=['forecast val'])
        test_forecast = pd.DataFrame(model.forecast(x_test, y_test), index=y_test.index[1:], columns=['forecast test'])

        path = os.path.join('gridsearch', self.dir, 'figures', 'best_predictions.pdf')
        ax = val_real.plot()
        test_real.plot(ax=ax)
        val_prediction.plot(ax=ax)
        test_prediction.plot(ax=ax)
        plt.ylabel(' '.join(self.dir.split('_')))
        plt.savefig(path)
        plt.show()

        path = os.path.join('gridsearch', self.dir, 'figures', 'best_forecasts.pdf')
        ax = val_real.plot()
        test_real.plot(ax=ax)
        val_forecast.plot(ax=ax)
        test_forecast.plot(ax=ax)
        plt.ylabel(' '.join(self.dir.split('_')))
        plt.savefig(path)
        plt.show()

    def save_results(self):
        directory = os.path.join('gridsearch', self.dir)
        if not os.path.exists(directory):
            os.mkdir(directory)

        csv_path = self.save_csv(directory)
        self.save_model(directory)

        path = os.path.join(directory, 'figures')
        if not os.path.exists(path):
            os.mkdir(path)
        GridResultsPlotter(csv_path, output_dir=path).plot_all()

    def save_csv(self, directory):
        csv_path = os.path.join(directory, 'results.csv')
        if os.path.exists(csv_path):
            old_res = pd.read_csv(csv_path, index_col=False)

            if set(old_res.columns) != set(self.parameter_results.columns):
                print 'Conflicting parameter results!'

            if len(list(self.parameter_results)) > len(list(old_res)):
                res = pd.concat([old_res, self.parameter_results], join_axes=[self.parameter_results.columns])
            else:
                res = pd.concat([old_res, self.parameter_results], join_axes=[old_res.columns])
            res.drop_duplicates(res.columns[:-2], inplace=True)
            # aggreagate duplicates??
            # print res
            # res.fillna('na', inplace=True)
            # gr = res.groupby(res.columns.tolist()[:-2])
            #
            # res = gr.agg(np.mean).reset_index()
            # res.replace('na', np.nan, inplace=True)
            # print res

        else:
            res = self.parameter_results
        res.to_csv(csv_path, index=False)
        return csv_path

    def save_model(self, directory):
        model_path = os.path.join(directory, 'best.json')
        perf_path = os.path.join(directory, 'perf.txt')

        flag = True
        if os.path.exists(perf_path):
            with open(perf_path, 'r') as f:
                old_perf = float(f.read())
            if old_perf < self.best_performance:
                flag = False

        if flag:
            self.best_estimator.model.save(model_path)
            with open(perf_path, 'w') as f:
                f.write(str(self.best_performance))


class GridResultsPlotter:
    def __init__(self, path, output_dir, show=False):
        self.show = False
        self.path = path
        self.out = output_dir
        self.data = pd.read_csv(path, index_col=False)

    def _remove_cols(self):
        cols_to_drop = [i for i in self.data.columns if len(self.data[i].drop_duplicates().dropna()) <= 1]
        return self.data.drop(cols_to_drop, axis=1)

    def pairplot(self):
        sns.set()
        data = self._remove_cols()
        sns.pairplot(data, hue='val', vars=data.columns[:-2],
                     palette=sns.color_palette('Blues_r'))

        path = os.path.join(self.out, 'pairplot.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def pairplot_val(self):
        sns.set()
        data = self._remove_cols()
        g = sns.pairplot(data, x_vars=data.columns[:-2], y_vars=data.columns[-2:])
        g.set(yscale='log')
        g.map(plt.plot)

        path = os.path.join(self.out, 'pairplot_val.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def parallel_coordinates(self):
        data = self._remove_cols()
        data = data.sort_values('val')
        colors = sns.color_palette('Blues_r', len(data))
        ax = parallel_coordinates(data, 'val', cols=data.columns[:-2], color=colors)
        ax.set_yscale('log')

        path = os.path.join(self.out, 'parallel_coordinates.pdf')
        plt.savefig(path)
        if self.show:
            plt.show()

    def cond_one_to_many(self):
        data = self._remove_cols()

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
            g = g.map(plt.plot, comb[0], 'val', color='r', label='val')
            g = g.map(plt.plot, comb[0], 'train', color='b', label='train')
            g.add_legend()

            path = os.path.join(self.out, 'conditional' + i + '.pdf')
            plt.savefig(path)
            if self.show:
                plt.show()

    def plot_all(self):
        try:
            self.pairplot()
        except Exception as e:
            print e
        try:
            self.pairplot_val()
        except Exception as e:
            print e
        try:
            self.parallel_coordinates()
        except Exception as e:
            print e
        try:
            self.cond_one_to_many()
        except Exception as e:
            print e
        try:
            self.cond_one_to_one()
        except Exception as e:
            print e
