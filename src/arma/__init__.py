import os
from statsmodels.tsa.arima_model import ARIMA
from src.utils import data_utils as du
from matplotlib import pyplot as plt
from src import preprocessing as pre
import pandas as pd
import numpy as np
import src.analysis as an
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import detrend
from sklearn.model_selection import train_test_split
from src.utils.experiments_utils import expr_sub_dir

EXP_DIR = expr_sub_dir('arma')


def main():
    country= 'EA'
    variable = 'CPI'
    data = du.get_data_dict()
    series = data[country][variable].dropna()


    # -----------------------------------
    # Plot Series
    # -----------------------------------
    series.plot()
    plt.legend()
    plt.show()



    # -----------------------------------
    # Preprocess
    # -----------------------------------
    # res = detrend(series)
    # res.plot()

    # res = seasonal_decompose(series.to_timestamp(), two_sided=False)
    # res.plot()
    # plt.show()
    # series.plot()
    # res.trend.plot(label='trend')
    # resid1 = series.to_timestamp() - res.trend
    # resid1.plot(label=' - trend')
    # res.seasonal.plot(label='seasonal')
    # res.resid.plot(label='residuals')
    # print an.stationary_tests(res.resid)
    # plt.legend()
    # plt.show()


    # resid, trend = pre.plot_trend_exp_smoothing(series, 2)
    # an.plot_rolling_stats(resid)
    # print an.stationary_tests(resid)
    # plt.show()
    # series = resid

    # resid, trend = pre.plot_trend_regression(series, 1)
    # an.plot_rolling_stats(resid)
    # print an.stationary_tests(resid)
    # plt.show()
    # series = resid

    # resid, trend = pre.plot_trend_smoothing(series)
    # an.plot_rolling_stats(resid)
    # plt.show()
    # print an.stationary_tests(resid)
    #
    # resid, seasonal = pre.plot_seasonal_regression(series,frequency=20, degree=20)
    # an.plot_rolling_stats(resid)
    # plt.show()
    # print an.stationary_tests(resid)
    #
    # resid = pre.plot_difference(series)
    # an.plot_rolling_stats(resid)
    # plt.show()
    # print an.stationary_tests(resid)

    # -----------------------------------
    # ACF and PACF
    # -----------------------------------

    an.plot_auto_correlation(series)
    plt.show()

    # -----------------------------------
    # Split train and test
    # -----------------------------------

    train, test = train_test_split(series, shuffle=False, test_size=0.2)
    plt.figure()

    # -----------------------------------
    # Fit ARIMA
    # -----------------------------------

    model = ARIMA(train, (3, 0, 0))
    fit = model.fit(disp=0)

    a = fit.predict()
    a.plot(label='prediction')
    series.plot()

    forecast, stderr, conf_int = fit.forecast(15)
    pd.Series(forecast, name='forecast', index=test.index).plot()
    pd.DataFrame(conf_int, columns=['Conf int1', 'conf int 2'], index=test.index).plot()
    plt.legend()
    plt.show()

    print(fit.summary())

    residuals = pd.DataFrame(fit.resid)
    residuals.plot()
    residuals.plot(kind='kde')

    plt.legend()
    plt.show()

    # figs = [plt.figure(n) for n in plt.get_fignums()]
    # for i, fig in enumerate(figs):
    #     fig.savefig(os.path.join(EXP_DIR, 'fig%d.png' % i))


if __name__ == '__main__':
    main()
