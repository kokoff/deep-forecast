import pandas as pd
from src.utils import data_utils
from statsmodels.tsa.stattools import adfuller, kpss
from itertools import product
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np


def stationary_test(series):
    res1 = 'not stationary'
    res2 = 'not stationary'

    adf_p = adfuller(series)[1]
    kpss_p = kpss(series)[1]

    if adf_p < 0.05:
        res1 = 'stationary'

    if kpss_p > 0.05:
        res2 = 'stationary'

    print 'ADF:', res1, '\t p=', adf_p
    print 'KPSS:', res2, '\t p=', kpss_p


class DifferenceTransformer(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.x_diff = None
        self.y_diff = None

    def fit(self, x, y):
        self.x = x.shift()
        self.y = y.shift()
        self.x_diff = x.diff().dropna()
        self.y_diff = y.diff().dropna()
        return x.iloc[1:], y.iloc[1:]

    def transform(self, x, y):
        x_trans = self.x_diff.reindex(x.index).dropna()

        y_trans = self.y_diff.reindex(y.index).dropna()

        return x_trans, y_trans

    def inverse_transform(self, y, y_true, recursive=False):
        if recursive:
            constant = self.y.reindex(y_true.index).iloc[0].values
            y_trans = np.cumsum(y, axis=1) + constant
            y_trans = y_trans.T
        else:
            y_trans = y + self.y.reindex(y_true.index)

        return y_trans


def main():
    data_params = OrderedDict()
    data_params['country'] = 'EA'
    data_params['vars'] = (['CPI', 'GDP'], ['CPI', 'GDP'])
    data_params['lags'] = (1, 1)

    tr = DifferenceTransformer()

    x, y = data_utils.get_data_in_shape(**data_params)
    x, y = tr.fit(x, y)
    x_train, x_val, x_test = data_utils.train_val_test_split(x, 12, 12)
    y_train, y_val, y_test = data_utils.train_val_test_split(y, 12, 12)


    curr_x = x_val
    cuur_y = y_val
    x_trans, y_trans = tr.transform(curr_x, cuur_y)
    y_orig = tr.inverse_transform(y_trans.values, cuur_y)

    print np.concatenate([cuur_y, y_orig], axis=1)
    print y_orig

    # print pd.concat([x_train, y_train], axis=1)
    # print pd.concat([x_trans, y_trans], axis=1)

    # plt.plot(y_train.values, 'r+', label='a')
    # print y_trans
    # inv_tr = tr.inverse_transform(y_trans.values, y_train)
    # plt.plot(inv_tr, 'b--', label='b')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
