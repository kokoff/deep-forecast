from inspect import getargspec

import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.activations import linear

from src.utils import data_utils


def split_input_variables(data):
    data_vars = []
    for column in data.columns.levels[0]:
        df = pd.DataFrame(data[column])
        data_vars.append(df)
    if len(data_vars) == 1:
        return data_vars[0]
    else:
        return data_vars


class ForecastModel(Model):

    @classmethod
    def InputLayerFromData(cls, x):
        variables = x.columns.levels[0].tolist()
        num_lags = len(x.columns.levels[1].tolist())
        if len(variables) == 1:
            inputs = Input(shape=(num_lags,), name='input_' + variables[0])
            layer = inputs
            return inputs, layer
        else:
            inputs = []
            for variable in variables:
                inputs.append(Input(shape=(num_lags,), name='input_' + variable))
            layer = concatenate(inputs)
            return inputs, layer

    @classmethod
    def OutputLayerFromData(cls, y, prev_layers):
        variables = y.columns.levels[0].tolist()
        num_lags = len(y.columns.levels[1].tolist())
        if len(variables) == 1:
            return Dense(1, activation=linear, name='output_' + variables[0])(prev_layers)
        else:
            outputs = []
            for variable in variables:
                outputs.append(Dense(num_lags, activation=linear, name='output_' + variable)(prev_layers))
            return outputs

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):
        x = split_input_variables(x)
        y = split_input_variables(y)
        validation_data = [split_input_variables(i) for i in validation_data]
        return super(ForecastModel, self).fit(x, y, batch_size, epochs, verbose, callbacks, validation_split,
                                              validation_data,
                                              shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                                              validation_steps, **kwargs)

    def predict(self, x, y, batch_size=None, verbose=0, steps=None):
        x = split_input_variables(x)
        prediction = pd.DataFrame(0.0, index=y.index, columns=y.columns)

        output = super(ForecastModel, self).predict(x, batch_size, verbose, steps)
        if isinstance(self.output, list):
            prediction.iloc[:, :] = np.concatenate(output, axis=1)
        else:
            prediction.iloc[:, :] = output

        return prediction

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
        x = split_input_variables(x)
        y = split_input_variables(y)
        return super(ForecastModel, self).evaluate(x, y, batch_size, verbose, sample_weight, steps)

    def forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        x_labels = x.columns.levels[0].tolist()
        y_labels = y.columns.levels[0].tolist()
        forecast = pd.DataFrame(0.0, index=y.index, columns=y.columns)
        x = split_input_variables(x)
        y = split_input_variables(y)

        if not isinstance(self.input, list) and not isinstance(self.output, list):
            x_lags = int(self.input.shape[1])
            x_vars = 1
            y_lags = int(self.output.shape[1])
            y_vars = 1
            data_len = len(x)
            x_matrix = x.as_matrix()

            assert (y_lags == 1)

            y_matrix = np.zeros((data_len, y_lags))
            input = x_matrix[0].reshape(1, x_lags)

            for i in range(data_len):
                output = super(ForecastModel, self).predict(input, batch_size, verbose, steps)
                # output = x_matrix[i + 1][0:1].reshape(1, y_lags)
                y_matrix[i] = output
                input = np.concatenate((output, input[:, :-y_lags]), axis=1)

            forecast.iloc[:, :] = y_matrix
            return forecast

        elif isinstance(self.input, list) and not isinstance(self.output, list):
            x_lags = int(self.input[0].shape[1])
            x_vars = len(self.input)
            y_lags = int(self.output.shape[1])
            y_vars = 1
            data_len = len(x[0])
            x_matrices = [i.as_matrix() for i in x]

            y_matrix = np.zeros((data_len, y_lags))
            input = [i[0].reshape(1, x_lags) for i in x_matrices]
            valid_indexes = [x_labels.index(i) for i in y_labels if i in x_labels]

            for i in range(data_len):
                new_input = [j[i].reshape(1, x_lags) for j in x_matrices]
                for index, var in enumerate(input):
                    if index in valid_indexes and i > 0:
                        new_input[index] = np.concatenate((output, var[:, :-y_lags]), axis=1)
                input = new_input

                output = super(ForecastModel, self).predict(input, batch_size, verbose, steps)
                # output = x_matrix[i + 1][0:1].reshape(1, y_lags)
                y_matrix[i] = output

            forecast.iloc[:, :] = y_matrix
            return forecast

        elif isinstance(self.input, list) and isinstance(self.output, list):
            x_lags = int(self.input[0].shape[1])
            x_vars = len(self.input)
            y_lags = int(self.output[0].shape[1])
            y_vars = len(self.output)

            data_len = len(x[0])
            x_matrices = [i.as_matrix() for i in x]

            y_matrix = [np.zeros((data_len, y_lags)) for i in range(y_vars)]
            input = [i[0].reshape(1, x_lags) for i in x_matrices]
            valid_indexes = [x_labels.index(i) for i in y_labels if i in x_labels]

            for i in range(data_len):
                new_input = [j[i].reshape(1, x_lags) for j in x_matrices]
                for index, var in enumerate(input):
                    if index in valid_indexes and i > 0:
                        new_input[index] = np.concatenate((output[index], var[:, :-y_lags]), axis=1)
                input = new_input

                output = super(ForecastModel, self).predict(input, batch_size, verbose, steps)
                # output = x_matrix[i + 1][0:1].reshape(1, y_lags)
                for j in range(y_vars):
                    y_matrix[j][i] = output[j]

            forecast.iloc[:, :] = np.concatenate(output, axis=1)
            return forecast

        else:
            raise ValueError('Output cannot be bigger than input')

    def evaluate_forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')

        forecast = self.forecast(x, y, batch_size=batch_size, verbose=verbose, steps=steps)

        if not isinstance(self.output, list):
            y_true = np.array(y.as_matrix())
            forecast = np.array(forecast.as_matrix())
            loss = self.loss(y_true, forecast)
            loss = K.eval(loss)
            losses = loss.mean()
        else:
            losses = [0]
            for column in forecast.columns.levels[0]:
                forc = np.array(forecast[column].as_matrix())
                y_true = np.array(y[column].as_matrix())
                loss = self.loss(y_true, forc)
                loss = K.eval(loss)
                losses.append(loss.mean())
            losses[0] = sum(losses)

        return losses


class ModelWrapper:
    def __init__(self, build_fn):
        self.build_fn = build_fn
        self.data_set = False
        self.param_set = False
        self.fitted = False

        self.input_size = None
        self.output_size = None

        self.params = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None

    def check(self):
        if not self.data_set:
            raise ValueError('Data has not been set!')
        if not self.param_set:
            raise ValueError('Params have not been set!')

    def filter_params(self, func):
        new_args = {}
        f_args = getargspec(func).args
        for key in self.params:
            if key in f_args:
                new_args[key] = self.params[key]
        return new_args

    def set_data(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.input_size = self.x_train.shape[1]
        self.output_size = self.y_train.shape[1]

        self.data_set = True
        if self.param_set:
            self.reset()

    def get_data(self):
        if not self.data_set:
            raise ValueError('Cannot get data. Data has not been set.')
        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

    def set_data_params(self, **data_params):
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data_utils.get_data_formatted(
            **data_params)
        self.input_size = self.x_train.shape[1]
        self.output_size = self.y_train.shape[1]

        self.data_set = True
        if self.param_set:
            self.reset()

    def set_params(self, **params):
        self.params = params
        self.param_set = True
        if self.data_set:
            self.reset()

    def reset(self):
        self.check()
        params = self.filter_params(self.build_fn)
        self.model = self.build_fn(self.x_train, self.y_train, **params)
        self.fitted = False

    def fit(self, verbose=False):
        self.check()
        params = self.filter_params(self.model.fit)
        self.fitted = True
        return self.model.fit(self.x_train, self.y_train, verbose=verbose, validation_data=[self.x_val, self.y_val],
                              **params)

    def predict(self, data, verbose=0):
        self.check()
        if not self.fitted:
            self.fit()

        if data is 'train':
            return self.model.predict(self.x_train, self.y_train, verbose=verbose)
        elif data is 'val':
            return self.model.predict(self.x_val, self.y_val, verbose=verbose)
        elif data is 'test':
            return self.model.predict(self.x_test, self.y_test, verbose=verbose)

    def evaluate(self, data):
        self.check()
        if not self.fitted:
            self.fit()

        if data is 'train':
            return self.model.evaluate(self.x_train, self.y_train)
        elif data is 'val':
            return self.model.evaluate(self.x_val, self.y_val)
        elif data is 'test':
            return self.model.evaluate(self.x_test, self.y_test)

    def forecast(self, data, verbose=0):
        self.check()
        if not self.fitted:
            self.fit()

        if data is 'train':
            return self.model.forecast(self.x_train, self.y_train, verbose=verbose)
        elif data is 'val':
            return self.model.forecast(self.x_val, self.y_val, verbose=verbose)
        elif data is 'test':
            return self.model.forecast(self.x_test, self.y_test, verbose=verbose)

    def evaluate_forecast(self, data):
        self.check()
        if not self.fitted:
            self.fit()

        if data is 'train':
            return self.model.evaluate_forecast(self.x_train, self.y_train)
        elif data is 'val':
            return self.model.evaluate_forecast(self.x_val, self.y_val)
        elif data is 'test':
            return self.model.evaluate_forecast(self.x_test, self.y_test)
