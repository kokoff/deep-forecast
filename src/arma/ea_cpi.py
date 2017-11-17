import os
from statsmodels.tsa.arima_model import ARIMA, ARMA
from src.utils import data_utils as du
from matplotlib import pyplot as plt
from src import preprocessing as pre
import pandas as pd
import numpy as np
import src.analysis as an
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import detrend
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.utils.experiments_utils import expr_sub_dir


def main():
    country = 'EA'
    variable = 'CPI'

    exp_dir = expr_sub_dir('arma', country + '_' + variable)
    block = True

    data = du.get_data()
    series = data[country][variable].dropna()

    # -----------------------------------
    # Plot Series
    # -----------------------------------
    # series.plot()
    # an.plot_rolling_stats(series)
    # print an.stationary_tests(series)
    # plt.legend()
    # plt.show(block=block)



    # -----------------------------------
    # Preprocess
    # -----------------------------------
    trend = 0
    seasonal = 0
    resid = None

    resid = pre.difference(series)
    plt.show(block=block)

    # -----------------------------------
    # ACF and PACF
    # -----------------------------------

    if resid is not None:
        an.plot_auto_correlation(resid)
    else:
        an.plot_auto_correlation(series)
    plt.show(block=block)

    # -----------------------------------
    # Split train and test
    # -----------------------------------

    train, test = train_test_split(series, shuffle=False, test_size=0.2)
    plt.figure()

    # -----------------------------------
    # Fit ARIMA
    # -----------------------------------

    model = ARMA(train, (2, 1), freq='Q')
    model_fit = model.fit(disp=0, full_output=True)
    print(model_fit.summary2())

    prediction = model_fit.predict(start=0, end=len(train) + (len(test)))
    model_fit.plot_predict(start=0, end=len(train) + (len(test)))
    series.plot()
    plt.legend()
    plt.show()

    residuals = pd.Series(model_fit.resid, index=train.index, name='arima residuals')

    # -----------------------------------
    # Post Process
    # -----------------------------------
    series += trend + seasonal
    prediction += trend + seasonal
    forecast += trend + seasonal
    conf_int += trend + seasonal

    # -----------------------------------
    # Plot
    # -----------------------------------

    print 'Test Recursive MSE', mean_squared_error(test, forecast)
    print 'Train One Step MSE', mean_squared_error(train[len(train) - len(prediction):], prediction)

    series.plot()
    prediction.plot(label='prediction')
    forecast.plot()
    conf_int['conf interval 1'].plot()
    conf_int['conf interval 2'].plot()
    plt.legend()
    plt.show(block=block)

    an.plot_rolling_stats(residuals)
    plt.legend()
    plt.figure()

    residuals.plot(kind='kde')
    plt.legend()
    plt.show(block=block)

    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(exp_dir, 'fig%d.png' % i))

    plt.show()


if __name__ == '__main__':
    main()
