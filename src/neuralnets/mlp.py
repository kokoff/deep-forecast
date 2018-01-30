from keras import losses, activations
from keras.layers import Input, Dense
from forecast_models import ForecastModel, GridSearch, ModelWrapper
from sklearn.model_selection import ParameterGrid
from src.utils import data_utils
from utils import get_xy_data
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np


def getModel(input_dim, output_dim, neurons):
    input = Input(shape=(input_dim,))
    layer = Dense(neurons, activation=activations.relu)(input)
    output = Dense(output_dim, activation=activations.linear)(layer)

    model = ForecastModel(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss=losses.mse)

    return model


def main():
    data = data_utils.get_ea_data(drop_na=True)
    X, Y = get_xy_data(data, 1, 1)

    x = X[['CPI']]
    y = Y[['CPI']]

    print x.as_matrix()

    x_train, x_val, x_test = data_utils.train_val_test_split(x, val_size=12, test_size=12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, val_size=12, test_size=12)

    # print adfuller(np.squeeze(x.as_matrix()))
    # print kpss(np.squeeze(x.as_matrix()))
    # x.plot()
    # plt.show()
    # sns.distplot(x_train)
    # plt.show()
    # sns.distplot(x_val)
    # plt.show()
    # sns.distplot(x_test)
    # plt.show()
    # print 'val-test', mannwhitneyu(x_val, x_test)
    # print  'train-val', mannwhitneyu(x_train, x_val)
    # print 'train-test', mannwhitneyu(x_train, x_test)
    #
    # print len(x_train), len(x_val), len(x_test)
    #
    # return
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    # print output_dim
    #
    # model = getModel(input_dim, output_dim, 7)
    # model.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=[x_val, y_val], shuffle=False)
    # print model.evaluate(x_test, y_test)
    # print model.evaluate_forecast(x_test, y_test)

    fit_params = dict(batch_size=10,
                      epochs=[100, 150, 200],
                      shuffle=False,
                      verbose=0)

    param_grid = ParameterGrid(dict(input_dim=[input_dim],
                                    output_dim=[output_dim],
                                    neurons=[5, 10, 15, 20]))

    grid_search = GridSearch(getModel, param_grid)
    best = grid_search.grid_search(x_train, y_train, x_val, y_val, **fit_params)

    print best.model.forecast(x_test, y_test)


if __name__ == '__main__':
    main()
