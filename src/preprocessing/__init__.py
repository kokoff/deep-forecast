import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from src.utils import data_utils as du
from statsmodels.tsa import seasonal as se
from src import analysis as an


def trend_regression(series, degree=1, plot=True):
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

    if plot:
        plt.figure()
        series.plot()
        resid.plot()
        trend.plot()
        plt.legend()

        print 'Stationary tests for', resid.name
        print an.stationary_tests(resid)
        an.plot_rolling_stats(resid)

    return resid, trend


def seasonal_regression(series, frequency=4, degree=10, plot=True):
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

    if plot:
        plt.figure()
        series.plot()
        resid.plot()
        seasonal.plot()
        plt.legend()

        print 'Stationary tests for', resid.name
        print an.stationary_tests(resid)
        an.plot_rolling_stats(resid)

    return resid, seasonal


def trend_smoothing(series, window=8, plot=True):
    du.remove_na(series)

    trend = series.rolling(window=window, center=True).mean()
    resid = series - trend

    # rename resulting series
    resid.name = series.name + ' smoothed'
    trend.name = series.name + ' trend smoothed'

    if plot:
        plt.figure()
        series.plot()
        resid.plot()
        trend.plot()
        plt.legend()

        print 'Stationary tests for', resid.name
        print an.stationary_tests(resid)
        an.plot_rolling_stats(resid)

    return resid, trend


def difference(series, offset=1, order=1, plot=True):
    du.remove_na(series)

    for i in range(order):
        shifted = series.shift(offset)
        resid = series - shifted

    # rename resulting series
    resid.name = series.name + ' differenced'

    if plot:
        plt.figure()
        series.plot()
        resid.plot()
        plt.legend()

        print 'Stationary tests for', resid.name
        print an.stationary_tests(resid)
        an.plot_rolling_stats(resid)

    return resid


def trend_exp_smoothing(series, com=1, plot=True):
    trend = series.ewm(com=com).mean()
    resid = series - trend

    # rename resulting series
    resid.name = series.name + ' exp smoothed'
    trend.name = series.name + ' trend exp smoothed'

    if plot:
        plt.figure()
        series.plot()
        resid.plot()
        trend.plot()
        plt.legend()

        print 'Stationary tests for', resid.name
        print an.stationary_tests(resid)
        an.plot_rolling_stats(resid)

    return resid, trend


def seasonal_decompose(series, plot=True):
    du.remove_na(series)

    res = se.seasonal_decompose(series.to_timestamp())

    res.trend.index = series.index
    res.trend.name = series.name + ' trend'
    res.seasonal.index = series.index
    res.seasonal.name = series.name + ' seasonal'
    res.resid.index = series.index
    res.resid.name = series.name + ' decomposed'

    if plot:
        res.plot()

        plt.figure()
        series.plot()
        res.trend.plot()
        resid1 = series - res.trend
        resid1.plot(label=series.name + ' - ' + res.trend.name)
        res.seasonal.plot()
        res.resid.plot()
        plt.legend()

        print 'Stationary tests for', res.resid.name
        print an.stationary_tests(res.resid)
        an.plot_rolling_stats(res.resid)

    return res.resid, res.trend, res.seasonal


def logarithm(series, min_value=None, plot=True):
    if min_value and series.min() < min_value:
        translation = min_value - series.min()
    else:
        translation = 0

    print series.min(), translation

    resid = np.log(series + translation)
    resid.name = series.name + ' log'



    if plot:
        plt.figure()
        series.plot()
        resid.plot()
        plt.legend()

        print 'Stationary tests for', resid.name
        print an.stationary_tests(resid)
        an.plot_rolling_stats(resid)

    return resid


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
