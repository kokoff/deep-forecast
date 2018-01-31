import pandas as pd
import numpy as np
from keras import losses, activations
from keras.layers import Input, Dense
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit, train_test_split, fit_grid_point
from sklearn.metrics import mean_squared_error, make_scorer
import os
from copy import deepcopy

from src.utils import data_utils
from src.utils.data_utils import get_xy_data
from json import dumps, loads


# Keras model with forecast and evaluate_forecast functions
# NOTE: evaluation may not work with multiple outputs
class ForecastModel(Model):
    # Recursively forecast on x (accepts any type of input output dimesnions)
    def forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        if not self.built:
            self.build()

        x_lags = len(x.columns.levels[1])
        y_lags = len(y.columns.levels[1])

        x_vars = [i for i in list(x.columns.levels[0]) if not x[[i]].empty]
        y_vars = [i for i in list(y.columns.levels[0]) if not y[[i]].empty]

        x_mat = np.copy(x.as_matrix())
        y_mat = np.copy(y.as_matrix())

        y_get_index = [y_vars.index(i) * y_lags + y_lags - 1 for i in x_vars if i in y_vars]
        y_set_index = [x_vars.index(i) * x_lags for i in x_vars if i in y_vars]
        x_set_index = [x_vars.index(i) * x_lags + j for i in x_vars for j in range(1, x_lags) if i in y_vars]
        x_get_index = [i - 1 for i in x_set_index if i > 0]
        next_get_index = [i for i in range(len(x_vars) * x_lags) if i not in x_set_index and i not in y_set_index]
        next_set_index = [i for i in range(len(x_vars) * x_lags) if i not in x_set_index and i not in y_set_index]

        def get_next_val(input, output, row):
            if len(y_get_index) > 0 and len(y_set_index) > 0:
                input[0][y_set_index] = output[0][y_get_index]
            input[0][x_set_index] = x_mat[row][x_get_index]
            input[0][next_set_index] = x_mat[row + 1][next_get_index]
            return input

        # Check for bugs
        input = np.copy(x_mat[0:1])
        input = get_next_val(input, y_mat, 0)
        assert (np.array_equal(input[0], x_mat[1]))

        forecast = np.copy(y_mat[1:])
        input = np.copy(x_mat[0:1])

        for i in range(len(x_mat) - 1):
            output = self.predict(input, batch_size=batch_size, verbose=verbose, steps=steps)
            if y_lags * len(y_vars) == 1:
                output = np.array([output])

            forecast[i] = output
            input = get_next_val(input, output, i)

        return forecast

    def evaluate_forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')

        y_true = np.array(y.as_matrix()[1:])
        forecast = self.forecast(x, y, batch_size=batch_size, verbose=verbose, steps=steps)

        loss = K.eval(self.loss(y_true, forecast)).mean()

        return loss


class ModelWrapper(KerasRegressor):

    # Recursively forecast on x (accepts any type of input output dimesnions)
    def forecast(self, x, y, **kwargs):
        kwargs = self.filter_sk_params(Model.predict, kwargs)
        return self.model.forecast(x, y, **kwargs)

    # Evaluate forecast MSE other losses not supported
    def score_forecast(self, x, y, **kwargs):
        kwargs = self.filter_sk_params(Model.evaluate, kwargs)
        loss = self.model.evaluate_forecast(x, y, **kwargs)
        if isinstance(loss, list):
            return -loss[0]
        return -loss


class GridSearch:
    class Best:
        def __init__(self):
            self.prediction = - np.inf
            self.forecast = - np.inf
            self.model = None
            self.params = None

    class Results:

        def __init__(self):
            self.average = None
            self.dir = None

        def create_result_dir(self):
            dir = raw_input('Enter directory for results:')
            dir = os.path.join('results', dir)
            while os.path.exists(dir):
                dir = raw_input('Directory exists please enter another:')
                dir = os.path.join('results', dir)

            os.mkdir(dir)
            self.dir = dir

        def init_logs(self, param_grid, num_runs, epochs):
            self.create_result_dir()
            names = param_grid[0].keys()
            index = [tuple(i.values()) for i in param_grid]
            index = pd.MultiIndex.from_tuples(index, names=names)

            self.average = pd.DataFrame(0.0, index=index, columns=['train', 'val'])

        def save_results(self):
            if not os.path.exists(self.dir):
                self.create_result_dir()

            self.average.to_csv(os.path.join(self.dir, 'results.csv'))

    def __init__(self, build_fn, param_grid, num_runs=20):
        self.build_fn = build_fn
        self.estimator = ModelWrapper(build_fn)
        self.param_grid = param_grid
        self.num_runs = num_runs

        self.results = self.Results()

        self.best = self.Best()

    def grid_search(self, x_train, y_train, x_val, y_val, **fit_params):

        self.results.init_logs(self.param_grid, self.num_runs, fit_params['epochs'])

        for params in self.param_grid:
            train = np.zeros(self.num_runs)
            val = np.zeros(self.num_runs)

            for i in range(self.num_runs):
                print 'Run ', i, '\tParameters ', str(params)
                self.estimator.set_params(**params)

                # Fit model
                self.estimator.fit(x_train, y_train, validation_data=[x_val, y_val], **fit_params)

                train[i] = self.estimator.score(x_train, y_train, verbose=0)
                val[i] = self.estimator.score(x_val, y_val, verbose=0)

            self.results.average.at[tuple(params.values()), 'train'] = train.mean()
            self.results.average.at[tuple(params.values()), 'val'] = val.mean()

            self.save_best(params)

        print self.results.average
        self.results.save_results()
        return self.best

    def save_best(self, params):
        avg_prediction = self.results.average.ix[tuple(params.values()), 'val']

        if avg_prediction > self.best.prediction:
            print avg_prediction
            self.best.model = ForecastModel.from_config(self.estimator.model.get_config())
            self.best.prediction = avg_prediction
            self.best.params = params



def test_forecast():
    for i, j in [(i, j) for i in range(1, 3) for j in range(1, 3)]:
        data = data_utils.get_ea_data(drop_na=True)
        X, Y = get_xy_data(data, i, j)

        for x_vars, y_vars in [(['CPI'], ['CPI']),
                               (['CPI', 'GDP'], ['CPI']),
                               (['CPI'], ['CPI', 'GDP']),
                               (['CPI', 'GDP'], ['CPI', 'GDP']),
                               (['CPI'], ['GDP']),
                               (['CPI', 'GDP'], ['CPI', 'UR'])]:
            x = X[x_vars]
            y = Y[y_vars]

            x_train, x_val, x_test = data_utils.train_val_test_split(x, val_size=0.15, test_size=0.15)
            y_train, y_val, y_test = data_utils.train_val_test_split(y, val_size=0.15, test_size=0.15)

            input_dim = x_train.shape[1]
            output_dim = y_train.shape[1]

            model = getModel(input_dim, output_dim, 7)
            model.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=[x_val, y_val], shuffle=False,
                      verbose=0)
            if model.evaluate_forecast(x_test, y_test) - model.evaluate_forecast(x_test, y_test) != 0:
                print y_vars, i, j
                print model.evaluate_forecast(x_test, y_test) - model.evaluate_forecast(x_test, y_test)
