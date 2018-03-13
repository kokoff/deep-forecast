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


def create_output_layers(num_outputs, prev_layers):
    if num_outputs == 1:
        return Dense(1, activation=linear)(prev_layers)
    else:
        outputs = []
        for i in range(num_outputs):
            outputs.append(Dense(1, activation=linear)(prev_layers))
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

        if isinstance(self.input, list):
            x_lags = int(self.input[0].shape[1])
            x_vars = len(self.input)
            x_matrix = np.array([i.as_matrix() for i in x])
        else:
            x_lags = int(self.input.shape[1])
            x_vars = 1
            x_matrix = np.reshape(x.as_matrix(), (x_vars, len(x), x_lags))

        if isinstance(self.output, list):
            y_lags = int(self.output[0].shape[1])
            y_vars = len(self.output)
            y_true_matrix = np.array([i.as_matrix() for i in y])
        else:
            y_lags = int(self.output.shape[1])
            y_vars = 1
            y_true_matrix = np.reshape(y.as_matrix(), (y_vars, len(y), y_lags))

        data_len = x_matrix.shape[1]
        y_matrix = np.zeros((y_vars, data_len, y_lags))

        input = x_matrix[:, 0:1]
        var_indexes = [x_labels.index(i) for i in y_labels if i in x_labels]

        for i in range(data_len):
            if i > 0:
                new_input = x_matrix[0:x_vars, i:i + 1].copy()
                old = input[var_indexes, :, :-y_lags]
                new_input[var_indexes] = np.concatenate([output, old], axis=-1)
                input = new_input

            output = super(ForecastModel, self).predict(list(input), batch_size, verbose, steps)
            output = np.reshape(output, (y_vars, 1, y_lags))

            # output = y_true_matrix[:, i:i + 1]
            y_matrix[:, i:i + 1] = output

            # print list(input)
            # print list(x_matrix[:, i:i + 1])

        y_matrix = np.squeeze(y_matrix)
        if y_vars == 1:
            y_matrix = np.expand_dims(y_matrix, -1)
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

    x, y = data_utils.get_data_in_shape('EA', (['CPI'], ['CPI']), 1)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input = Input(shape=(1,))
    layer = Dense(1)(input)

    model = ForecastModel(inputs=input, outputs=layer)
    model.compile(optimizer='adam', loss='mse')

    print model.forecast(x_val, y_val)


def many_one():
    from src.utils import data_utils
    from keras import layers

    x, y = data_utils.get_data_in_shape('EA', (['CPI', 'GDP'], ['CPI']), 2)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input1 = Input(shape=(2,))
    input2 = Input(shape=(2,))

    layer = layers.concatenate([input1, input2])
    layer = Dense(1)(layer)

    model = ForecastModel(inputs=[input1, input2], outputs=layer)
    model.compile(optimizer='adam', loss='mse')

    print model.forecast(x_val, y_val)


def many_many():
    from src.utils import data_utils
    from keras import layers

    x, y = data_utils.get_data_in_shape('EA', (['CPI', 'GDP'], ['CPI', 'GDP']), 6)
    x_train, x_val, x_test, = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)

    input1 = Input(shape=(6,))
    input2 = Input(shape=(6,))

    layer = layers.concatenate([input1, input2])
    layer1 = Dense(1)(layer)
    layer2 = Dense(1)(layer)

    model = ForecastModel(inputs=[input1, input2], outputs=[layer1, layer2])
    model.compile(optimizer='adam', loss='mse')

    print model.forecast(x_val, y_val)


def main1():
    pass

    # from src.utils import data_utils
    # data = data_utils.get_ea_data()[['CPI']]
    # print data
    # x, y = data_utils.get_xy_data(data, lags=2, lags2=2)
    # print pd.concat([x, y], axis=1)


if __name__ == '__main__':
    # one_one()
    # many_one()
    many_many()
    # main()
