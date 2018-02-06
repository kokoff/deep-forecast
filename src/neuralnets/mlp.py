from keras import losses, activations
from keras.layers import Input, Dense
from forecast_models import ForecastModel, ModelWrapper
from src.neuralnets.grid_search import GridSearch
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
from keras.models import model_from_json
import pandas as pd


def getMLP(input_dim, output_dim, neurons, optimizer='adam'):
    input = Input(shape=(input_dim,))
    layer = Dense(neurons, activation=activations.relu)(input)
    output = Dense(output_dim, activation=activations.linear)(layer)

    model = ForecastModel(inputs=input, outputs=output)

    model.compile(optimizer=optimizer, loss=losses.mse)

    return model


def main():
    parameters = OrderedDict()
    parameters['batch_size'] = [6, 10]
    parameters['epochs'] = [50, 100, 150, 200]
    parameters['shuffle'] = [False]
    parameters['verbose'] = [0]
    parameters['neurons'] = [2, 4, 6, 8, 10]

    data_dict = OrderedDict()
    data_dict['country'] = ['EA', 'US']
    data_dict['var_dict'] = [
        dict(x=['CPI'], y=['CPI']),
        dict(x=['GDP'], y=['GDP']),
        dict(x=['UR'], y=['UR']),
        dict(x=['IR'], y=['IR']),
        dict(x=['LR10'], y=['LR10']),
        dict(x=['LR10-IR'], y=['LR10-IR']),
        dict(x=['EXRATE'], y=['EXRATE'])
    ]
    data_dict['x_lag'] = [1]
    data_dict['y_lag'] = [1]
    data_dict['val_size'] = [12]
    data_dict['test_size'] = [12]

    # x_train, y_train, x_val, y_val, x_test, y_test = data_utils.get_data_formatted('EA', {'x': ['CPI'], 'y': ['CPI']},
    #                                                                                1, 1,
    #                                                                                12, 12)

    grid_search = GridSearch(getMLP, parameters, directory='CPI')
    grid_search.grid_search_data(data_dict)
    # model = grid_search.grid_search(x_train, y_train, x_val, y_val)
    # model = getMLP(1, 1, 1)
    # model.fit(x_train, y_train)

    # data = {'val prediction': model.evaluate(x_val, y_val),
    #         'val forecast': model.evaluate_forecast(x_val, y_val),
    #         'test prediction': model.evaluate(x_test, y_test),
    #         'test forecast': model.evaluate_forecast(x_test, y_test)}
    #
    # df = pd.DataFrame(data=data, index=[0])
    # print df
    #
    # plot_model(model, x_val, y_val, x_test, y_test)


def plot_model(model, x_val, y_val, x_test, y_test):
    val_real = pd.DataFrame(y_val.values, index=y_val.index, columns=['real val'])
    test_real = pd.DataFrame(y_test.values, index=y_test.index, columns=['real test'])
    val_prediction = pd.DataFrame(model.predict(x_val), index=y_val.index, columns=['prediction val'])
    test_prediction = pd.DataFrame(model.predict(x_test), index=y_test.index, columns=['prediction test'])
    val_forecast = pd.DataFrame(model.forecast(x_val, y_val), index=y_val.index[1:], columns=['forecast val'])
    test_forecast = pd.DataFrame(model.forecast(x_test, y_test), index=y_test.index[1:], columns=['forecast test'])

    ax = val_real.plot()
    test_real.plot(ax=ax)
    val_prediction.plot(ax=ax)
    test_prediction.plot(ax=ax)
    plt.show()

    ax = val_real.plot()
    test_real.plot(ax=ax)
    val_forecast.plot(ax=ax)
    test_forecast.plot(ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
