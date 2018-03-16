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


def split_data(data, splits):
    if isinstance(data, pd.DataFrame):
        data = data.values

    if splits > 1:
        return np.split(data, splits, axis=1)
    else:
        return data


class ForecastModel(Model):
    def __init__(self, inputs, outputs):
        super(ForecastModel, self).__init__(inputs, outputs)

        if isinstance(self.input, list):
            self.x_lags = int(self.input[0].shape[1])
            self.x_vars = len(self.input)
        else:
            self.x_lags = int(self.input.shape[1])
            self.x_vars = 1

        if isinstance(self.output, list):
            self.y_lags = int(self.output[0].shape[1])
            self.y_vars = len(self.output)
        else:
            self.y_lags = int(self.output.shape[1])
            self.y_vars = 1

        if self.y_vars > self.x_vars:
            raise ValueError('Input variables need to be more than output variables.')
        if self.y_lags > self.x_lags:
            raise ValueError('Input lags need to be more than output lags.')

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=0, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        # Split data if multi input/output model
        x = split_data(x, self.x_vars)
        y = split_data(y, self.y_vars)

        if validation_data:
            x_val = split_data(validation_data[0], self.x_vars)
            y_val = split_data(validation_data[1], self.y_vars)
            validation_data = [x_val, y_val]

        return super(ForecastModel, self).fit(x, y, batch_size, epochs, verbose, callbacks, validation_split,
                                              validation_data,
                                              shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                                              validation_steps, **kwargs)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        # Split data if multi input/output model
        x = split_data(x, self.x_vars)

        y = super(ForecastModel, self).predict(x, batch_size, verbose, steps)

        y = np.reshape(y, (self.y_vars, -1, self.y_lags))
        y = np.concatenate(list(y), axis=1)
        return np.array(y)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=0, sample_weight=None, steps=None):
        x = split_data(x, self.x_vars)
        y = split_data(y, self.y_vars)
        return super(ForecastModel, self).evaluate(x, y, batch_size, verbose, sample_weight, steps)

    def forecast(self, x, y, batch_size=None, verbose=0, steps=None):
        x = split_data(x, self.x_vars)
        y = split_data(y, self.y_vars)

        x_matrix = np.reshape(x, (self.x_vars, -1, self.x_lags))
        y_true_matrix = np.reshape(y, (self.y_vars, -1, self.y_lags))

        data_len = x_matrix.shape[1]
        y_matrix = np.zeros((self.y_vars, data_len, self.y_lags))

        input = x_matrix[:, 0:1]

        var_indexes = []
        for i in range(self.y_vars):
            for j in range(self.x_vars):
                if np.all(x_matrix[j, self.y_lags:, 0] == y_true_matrix[i, :-self.y_lags, 0]):
                    var_indexes.append(j)

        for i in range(data_len):
            if i > 0:
                new_input = x_matrix[0:self.x_vars, i:i + 1].copy()
                old = input[var_indexes, :, :-self.y_lags]
                new_input[var_indexes] = np.concatenate([output, old], axis=-1)
                input = new_input

            output = super(ForecastModel, self).predict(list(input), batch_size, verbose, steps)
            output = np.reshape(output, (self.y_vars, 1, self.y_lags))

            # output = y_true_matrix[:, i:i + 1]
            y_matrix[:, i:i + 1] = output

            # print list(input)
            # print list(x_matrix[:, i:i + 1])

        y_matrix = np.concatenate(list(y_matrix), axis=1)
        return y_matrix

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


class RecurrentModel(ForecastModel):
    def fit(self, x=None, y=None, batch_size=1, epochs=1, verbose=0, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):
        for i in range(epochs):
            super(RecurrentModel, self).fit(x=x, y=y, batch_size=batch_size, epochs=1, verbose=verbose,
                                            callbacks=callbacks, validation_split=validation_split,
                                            validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
                                            sample_weight=sample_weight, initial_epoch=initial_epoch,
                                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                            **kwargs)
            self.reset_states()

    def predict(self, x, batch_size=1, verbose=0, steps=None):
        return super(RecurrentModel, self).predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps)

    def forecast(self, x, y, batch_size=1, verbose=0, steps=None):
        return super(RecurrentModel, self).forecast(x=x, y=y, batch_size=batch_size, verbose=verbose, steps=steps)


def main():
    from src.utils import data_utils
    from keras import layers
    from matplotlib import pyplot as plt

    x, y = data_utils.get_data_in_shape('EA', (['CPI'], ['CPI']), 1)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input = Input(batch_shape=(1, 1,))
    layer = layers.Reshape((1, 1))(input)
    layer = layers.LSTM(10, stateful=True)(layer)
    # layer = layers.LSTM(10, return_sequences=False, stateful=True)(layer)
    layer = layers.Dense(1, activation='linear')(layer)

    model = RecurrentModel(inputs=input, outputs=layer)
    model.compile(optimizer='adam', loss='mse')

    model.fit(x_train, y_train, epochs=200, batch_size=1)
    pred = model.predict(x_val, batch_size=1)
    fcast = model.forecast(x_val, y_val, batch_size=1)

    from keras.utils.vis_utils import plot_model

    plot_model(model)
    model.summary()

    plt.plot(y_val.values)
    plt.plot(pred)
    plt.show()

    plt.plot(y_val.values)
    plt.plot(fcast)
    plt.show()


def one_one():
    from src.utils import data_utils
    from keras import layers

    x, y = data_utils.get_data_in_shape('EA', (['CPI'], ['CPI']), 2, 2)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input = Input(shape=(2,))
    layer = Dense(2)(input)

    model = ForecastModel(inputs=input, outputs=layer)
    model.compile(optimizer='adam', loss='mse')

    print 'one one'
    model.fit(x_train, y_train)
    print model.predict(x_val).shape
    print model.forecast(x_val, y_val).shape


def m_m_m():
    from src.utils import data_utils
    from keras import layers

    x, y = data_utils.get_data_in_shape('EA', (['CPI', 'GDP'], ['CPI', 'GDP']), 2, 2)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input1 = Input(shape=(2,))
    input2 = Input(shape=(2,))
    conc = layers.concatenate([input1, input2])

    layer = Dense(2)(conc)
    layer1 = Dense(2)(conc)

    model = ForecastModel(inputs=[input1, input2], outputs=[layer, layer1])
    model.compile(optimizer='adam', loss='mse')

    print y_val
    fcast = model.forecast(x_val, y_val)
    print fcast
    print pd.DataFrame(list(fcast), index=y_val.index, columns=y_val.columns)


def many_one():
    from src.utils import data_utils
    from keras import layers

    data_params1 = {}
    data_params1['country'] = 'EA'
    data_params1['vars'] = (['CPI', 'GDP', 'UR'], ['CPI', 'GDP', 'UR'])
    data_params1['x_lags'] = 5
    data_params1['y_lags'] = 2

    xlags = data_params1['x_lags']
    ylags = data_params1['y_lags']
    xvars = len(data_params1['vars'][0])
    yvars = len(data_params1['vars'][1])

    x, y = data_utils.get_data_in_shape(**data_params1)

    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    inputs, layer = create_input_layers(xvars, xlags)

    outputs = create_output_layers(yvars, ylags, layer)

    model = ForecastModel(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    print 'many one'
    model.fit(x_train, y_train)
    print model.predict(x_val).shape
    print model.forecast(x_val, y_val).shape


def many_many():
    from src.utils import data_utils
    from keras import layers

    x, y = data_utils.get_data_in_shape('EA', (['CPI', 'GDP'], ['CPI', 'GDP']), 2, 2)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input1 = Input(shape=(2,))
    input2 = Input(shape=(2,))

    layer = layers.concatenate([input1, input2])
    layer1 = Dense(2)(layer)
    layer2 = Dense(2)(layer)

    model = ForecastModel(inputs=[input1, input2], outputs=[layer1, layer2])
    model.compile(optimizer='adam', loss='mse')

    print 'many many'
    model.fit(x_train, y_train)
    print model.predict(x_val).shape
    print model.forecast(x_val, y_val).shape


def main1():
    pass

    # from src.utils import data_utils
    # data = data_utils.get_ea_data()[['CPI']]
    # print data
    # x, y = data_utils.get_xy_data(data, lags=2, lags2=2)
    # print pd.concat([x, y], axis=1)


def valid_indexes(x, y):
    x_vars = x.shape[1]
    y_vars = y.shape[1]
    print x_vars, y_vars


def lag(data, lags, offset=0):
    cols = data.shape[-1]

    # check if pandas
    if isinstance(data, pd.DataFrame):
        matrix = data.values
    else:
        matrix = data

    # repeat each column
    matrix = np.repeat(matrix, lags, axis=1)

    # lag repeated columns
    for i in range(matrix.shape[1]):
        matrix[:, [i]] = np.roll(matrix[:, [i]], (i / cols), axis=0)

    # shift by lagged values
    matrix = matrix[lags - 1:]

    # shift by offset
    if offset > 0:
        matrix = matrix[offset - 1:]
    elif offset < 0:
        matrix = matrix[:offset + 1]

    return matrix


def lag_xy(x, y, x_lags, y_lags, dataframe=False):
    x_lagged = lag(x, x_lags, -y_lags)
    y_lagged = lag(y, y_lags, x_lags)

    if dataframe:
        columns = pd.MultiIndex.from_product((x.columns, ['x' + str(i) for i in range(x_lags)]))
        index = x.index[x_lags - 1: -y_lags + 1]
        x_lagged = pd.DataFrame(x_lagged, index=index, columns=columns)

        columns = pd.MultiIndex.from_product((y.columns, ['y' + str(i) for i in range(y_lags)]))
        index = y.index[y_lags - 1 + x_lags - 1:]
        y_lagged = pd.DataFrame(y_lagged, index=index, columns=columns)

    return x_lagged, y_lagged



if __name__ == '__main__':
    # x_lags = 2
    # y_lags = 2
    # x, y = data_utils.get_data_in_shape('EA', (['CPI', 'GDP'], ['CPI']), x_lags, y_lags)
    #
    # x = split_data(x, 2)
    # y = split_data(y, 1

    # valid_indexes(x.values, y.values)

    one_one()
    many_one()
    many_many()
    # main()
    # m_m_m()
