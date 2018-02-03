import pandas as pd
import numpy as np
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from sklearn.model_selection import ParameterGrid

from src.utils import data_utils
from src.utils.data_utils import get_xy_data, get_data_formatted


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

    def score(self, x, y, **kwargs):
        return - super(ModelWrapper, self).score(x, y, **kwargs)

    def forecast(self, x, y, **kwargs):
        return self.model.forecast(x, y, **kwargs)

    # Evaluate forecast MSE other losses not supported
    def score_forecast(self, x, y, **kwargs):
        loss = self.model.evaluate_forecast(x, y, **kwargs)

        if isinstance(loss, list):
            return loss[0]
        return loss


class GridSearch:

    def __init__(self, build_fn, param_dict, data_dict, num_runs=20):
        self.reset_logs(param_dict, data_dict)
        self.estimator = ModelWrapper(build_fn)
        self.param_grid = ParameterGrid(param_dict)
        self.data_params = ParameterGrid(data_dict)
        self.num_runs = num_runs

    def reset_logs(self, param_dict, data_disct):
        columns = ['train prediction', 'val prediction', 'train forecast', 'val forecast']
        self.run_results = pd.DataFrame(columns=columns)

        columns = pd.MultiIndex.from_product([['train', 'val'], ['prediction', 'forecast']])
        columns = data_disct.keys() + param_dict.keys() + ['input_dim', 'output_dim', 'train prediction',
                                                           'val prediction', 'train forecast', 'val forecast']
        self.parameter_results = pd.DataFrame(columns=columns)

        self.data_results = pd.DataFrame(columns=columns + ['test prediction', 'test forecast'])

    def grid_search(self):

        # self.reset_logs()

        for k, data_param in enumerate(self.data_params):
            x_train, y_train, x_val, y_val, x_test, y_test = get_data_formatted(**data_param)
            print data_param

            for j, params in enumerate(self.param_grid):
                params['input_dim'] = x_train.shape[1]
                params['output_dim'] = y_train.shape[1]

                for i in range(self.num_runs):
                    print 'Run ', i, '\tParameters ', str(params)
                    self.estimator.set_params(**params)

                    # Fit model
                    self.estimator.fit(x_train, y_train, validation_data=[x_val, y_val])

                    self.evaluate_run(x_train, y_train, x_val, y_val, i)

                self.evaluate_parameters(params, data_param, j, k)
                self.parameter_results.to_csv('params.csv')
            self.evaluate_data(x_test, y_test, data_param, k)
            self.data_results.to_csv('best.csv')

    def evaluate_run(self, x_train, y_train, x_val, y_val, run_num):
        self.run_results.at[run_num, 'train prediction'] = self.estimator.score(x_train, y_train)
        self.run_results.at[run_num, 'val prediction'] = self.estimator.score_forecast(x_train, y_train)
        self.run_results.at[run_num, 'train forecast'] = self.estimator.score(x_val, y_val)
        self.run_results.at[run_num, 'val forecast'] = self.estimator.score_forecast(x_val, y_val)
        self.run_results.to_csv('params.csv')

    def evaluate_parameters(self, params, data_params, param_index, data_index):
        index = data_index * len(self.param_grid) + param_index
        self.parameter_results.at[index, data_params.keys()] = data_params.values()
        self.parameter_results.at[index, params.keys()] = params.values()

        cols = ['train prediction', 'val prediction', 'train forecast', 'val forecast']
        self.parameter_results.at[index, cols] = self.run_results.mean().values

    def evaluate_model(self, x_val, y_val):
        if self.best_model_prediction > self.estimator.score(x_val, y_val):
            pass

    def evaluate_data(self, x_test, y_test, data_params, data_index):
        from operator import and_
        var = reduce(and_, [self.parameter_results[i] == j for i, j in data_params.iteritems()])
        max_index = self.parameter_results[var]['val prediction'].astype('float64').idxmin()
        self.data_results.at[data_index, :] = self.parameter_results.iloc[max_index]

        self.data_results.at[data_index, 'test prediction'] = self.estimator.score(x_test, y_test)
        self.data_results.at[data_index, 'test forecast'] = self.estimator.score_forecast(x_test, y_test)


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
