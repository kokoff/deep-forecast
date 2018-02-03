from keras import losses, activations
from keras.layers import Input, Dense
from forecast_models import ForecastModel, GridSearch, ModelWrapper
from sklearn.model_selection import ParameterGrid
from src.utils import data_utils
from src.utils.data_utils import get_xy_data, get_data_formatted
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
from collections import OrderedDict
import seaborn as sns

def getMLP(input_dim, output_dim, neurons, optimizer='adam'):
    input = Input(shape=(input_dim,))
    layer = Dense(neurons, activation=activations.relu)(input)
    output = Dense(output_dim, activation=activations.linear)(layer)

    model = ForecastModel(inputs=input, outputs=output)

    model.compile(optimizer=optimizer, loss=losses.mse)

    return model


def main():
    # data = data_utils.get_ea_data(drop_na=True)
    # X, Y = get_xy_data(data, 1, 1)
    #
    # x = X[['CPI']]
    # y = Y[['CPI']]
    #
    # x_train, x_val, x_test = data_utils.train_val_test_split(x, val_size=12, test_size=12)
    # y_train, y_val, y_test = data_utils.train_val_test_split(y, val_size=12, test_size=12)
    #
    # input_dim = x_train.shape[1]
    # output_dim = y_train.shape[1]

    parameters = OrderedDict()
    parameters['batch_size'] = [10]
    parameters['epochs'] = [100]
    parameters['shuffle'] = [False]
    parameters['verbose'] = [0]
    parameters['neurons'] = [2, 4]

    data_dict = OrderedDict()
    data_dict['country'] = ['EA', 'US']
    data_dict['var_list'] = [
        dict(x=['CPI'], y=['CPI']),
        # dict(x=['GDP'], y=['GDP']),
        # dict(x=['UR'], y=['UR']),
        # dict(x=['IR'], y=['IR']),
        # dict(x=['LR10'], y=['LR10']),
        # dict(x=['LR10-IR'], y=['LR10-IR']),
        # dict(x=['EXRATE'], y=['EXRATE'])
    ]
    data_dict['x_lag'] = [1]
    data_dict['y_lag'] = [1]
    data_dict['val_size'] = [12]
    data_dict['test_size'] = [12]

    grid_search = GridSearch(getMLP, parameters, data_dict, num_runs=2)
    grid_search.grid_search()

    parameters = OrderedDict()
    parameters['batch_size'] = 10
    parameters['epochs'] = 100
    parameters['shuffle'] = False
    parameters['verbose'] = 1
    parameters['neurons'] = 8
    parameters['input_dim'] = 1
    parameters['output_dim'] = 1

    m = ModelWrapper(getMLP, **parameters)
    x_train, y_train, x_val, y_val, x_test, y_test = get_data_formatted('EA', {'x': 'CPI', 'y': 'CPI'}, 1, 1, 12, 12)

    m.fit(x_train, y_train, validation_data=[x_val, y_val])

    import pandas as pd
    # print m.predict(x_val)
    print y_val.index.to_native_types()
    print y_test.index.to_native_types()
    plt.plot(y_val.index.to_native_types(), y_val.values, label='val targets')
    plt.plot(y_val.index.to_native_types(), m.predict(x_val), label='val prediction')
    plt.plot(y_test.index.to_native_types(), y_test.values, label='test targets')
    plt.plot(y_test.index.to_native_types(), m.predict(x_test), label='test prediction')
    plt.legend()
    plt.show()

    print np.squeeze(y_test.as_matrix()) - m.predict(x_test)

    sns.distplot(np.squeeze(y_test.as_matrix()) - m.predict(x_test))

    # plt.plot(y_val.index.to_native_types(), y_val.values, label='val targets')
    # plt.plot(y_val.index.to_native_types(), m.forecast(x_val, y_val), label='val forecast')
    # plt.plot(y_test.index.to_native_types(), y_test.values, label='test targets')
    # plt.plot(y_test.index.to_native_types(), m.forecast(x_test, y_test), label='test forecast')
    # plt.legend()
    plt.show()

    # plt.plot(y_val.values)
    # plt.plot(m.forecast(x_val))
    # plt.show()


if __name__ == '__main__':
    main()
