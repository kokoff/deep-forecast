from inspect import getargspec

import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.activations import linear

from src.utils import data_utils
from sklearn.model_selection import TimeSeriesSplit


def create_input_layers(num_inputs, input_size):
    if num_inputs == 1:
        inputs = Input(shape=(input_size,))
        return inputs, inputs
    else:
        inputs = []
        for i in range(num_inputs):
            inputs.append(Input(shape=(input_size,)))
        layer = concatenate(inputs)
        return inputs, layer


def create_output_layers(num_outputs, output_size, prev_layers):
    if num_outputs == 1:
        return Dense(output_size, activation=linear)(prev_layers)
    else:
        outputs = []
        for i in range(num_outputs):
            outputs.append(Dense(output_size, activation=linear)(prev_layers))
        return outputs


class ForecastModel(Model):

    def _split_data(self, x, y=None):
        if isinstance(self.input, list):
            x = np.split(x, len(self.input), axis=1)
        if y is not None and isinstance(self.output, list):
            y = np.split(y, len(self.output), axis=1)
        if y is not None:
            return x, y
        else:
            return x

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=0, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        # Split data if multi input/output model
        x, y = self._split_data(x, y)

        if validation_data:
            x_val, y_val = self._split_data(validation_data[0], validation_data[1])
            validation_data = [x_val, y_val]

        return super(ForecastModel, self).fit(x, y, batch_size, epochs, verbose, callbacks, validation_split,
                                              validation_data,
                                              shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                                              validation_steps, **kwargs)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        # Split data if multi input/output model
        x = self._split_data(x)

        return super(ForecastModel, self).predict(x, batch_size, verbose, steps)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=0, sample_weight=None, steps=None):
        x, y = self._split_data(x, y)
        return super(ForecastModel, self).evaluate(x, y, batch_size, verbose, sample_weight, steps)

    def forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        x_labels = x.columns.levels[0].tolist()
        y_labels = y.columns.levels[0].tolist()

        x, y = self._split_data(x, y)

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

            return np.array(y_matrix)

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

            return np.array(y_matrix)

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

            return np.squeeze(y_matrix)

        else:
            raise ValueError('Output cannot be bigger than input')

    def evaluate_forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')

        forecast = self.forecast(x, y, batch_size=batch_size, verbose=verbose, steps=steps)

        if not isinstance(self.output, list):
            y_true = np.array(y.as_matrix())
            forecast = np.array(forecast)

            loss = self.loss_functions[0](y_true, forecast)
            loss = K.eval(loss)
            losses = loss.mean()
        else:
            losses = [0]
            for index, column in enumerate(y.columns.levels[0]):
                forc = np.array(forecast[index])
                y_true = np.array(y[column].as_matrix())
                loss = self.loss_functions[index](y_true, forc)
                loss = K.eval(loss)
                losses.append(loss.mean())
            losses[0] = sum(losses)

        return losses
