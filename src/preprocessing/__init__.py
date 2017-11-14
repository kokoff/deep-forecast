import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from code.utils import data_utils as du


def trend_regression(series, degree=1):
    du.remove_na(series)

    x = np.array([i for i in range(1, len(series) + 1)])
    X = np.array([x ** i for i in range(1, degree + 1)]).T
    Y = series

    lm = LinearRegression()
    lm.fit(X, Y)

    trend = pd.Series(lm.predict(X), index=series.index)
    resid = Y - trend

    # rename resulting series
    resid.name = series.name + ' detrended'
    trend.name = series.name + ' trend regression'

    return resid, trend


def seasonal_regression(series, frequency=4, degree=10):
    du.remove_na(series)

    x = np.array([i % frequency for i in range(1, len(series) + 1)])
    x1 = np.array([i for i in range(1, len(series) + 1)])
    X = np.array(x1 + [x ** i for i in range(1, degree + 1)]).T
    Y = series

    lm = LinearRegression()
    lm.fit(X, Y)

    seasonal = pd.Series(lm.predict(X), index=series.index)
    resid = Y - seasonal

    # rename resulting series
    resid.name = series.name + ' seasonally adjusted'
    seasonal.name = series.name + ' seasonal regression'

    return resid, seasonal


def trend_smoothing(series, window=8):
    du.remove_na(series)

    trend = series.rolling(window=window, center=True).mean()
    resid = series - trend

    # rename resulting series
    resid.name = series.name + ' smoothed'
    trend.name = series.name + ' trend smoothed'

    return resid, trend


def difference(series, offset=1, order=1):
    du.remove_na(series)

    for i in range(order):
        shifted = series.shift(offset)
        series = series - shifted

    # rename resulting series
    series.name = series.name + ' differenced'

    return series


def main():
    data = du.get_us_data()
    series = data['LR10-IR']
    # resid, trend = trend_regression(series)
    series = np.log(series)

    resid, trend = trend_regression(series)
    series.plot()
    resid.plot()
    trend.plot()
    plt.legend()
    plt.show()

    resid, trend = seasonal_regression(series)
    series.plot()
    resid.plot()
    trend.plot()
    plt.legend()
    plt.show()

    resid, trend = trend_smoothing(series)
    series.plot()
    resid.plot()
    trend.plot()
    plt.legend()
    plt.show()

    resid = difference(series)
    series.plot()
    resid.plot()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
