import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasRegressor

from src.utils import data_utils
from src.utils.data_utils import get_xy_data
from keras.models import load_model
from keras.models import clone_model
import os
from copy import deepcopy


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

        loss = self.loss(y_true, forecast)
        loss = K.eval(loss)
        loss = loss.mean()

        return loss

    def copy(self):
        # self.save('.temp')
        # new_model = load_model('.temp', {'ForecastModel': ForecastModel})
        # os.remove('.temp')
        new_model =clone_model(self)
        return new_model

    def save_json(self, filename):
        json_str = self.to_json()
        with open(filename, 'w') as f:
            f.write(json_str)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            json_string = f.read()
        config = model_from_json(json_string, {'ForecastModel': ForecastModel}).get_config()
        return ForecastModel.from_config(config)


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


if __name__ == '__main__':
    from mlp import getMLP

    x_train, y_train, x_val, y_val, x_test, y_test = data_utils.get_data_formatted('EA', {'x': 'CPI', 'y': 'CPI'}, 2, 2,
                                                                                   12, 12)
    model = getMLP(x_train.shape[1], y_train.shape[1], 3)
    model.fit(x_train, y_train, verbose=False)

    model.save('temp')
    from keras.models import load_model

    nm = model.copy()

    print model.evaluate(x_val, y_val)
    print nm.evaluate(x_val, y_val)

    model.save_json('model.json')
    nm = ForecastModel.from_json('model.json')

    print model.evaluate(x_val, y_val)
