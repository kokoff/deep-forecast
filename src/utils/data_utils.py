import pandas as pd
import os
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data'))
DATA_PATH_XLS = os.path.join(DATA_PATH, 'Data_ILP.xls')
DATA_PATH_CSV_EA = os.path.join(DATA_PATH, 'EA.csv')
DATA_PATH_CSV_US = os.path.join(DATA_PATH, 'US.csv')

COUNTRIES = ['EA', 'US']
VARIABLES = ['CPI', 'GDP', 'UR', 'IR', 'LR10', 'LR10-IR', 'EXRATE']

VAL_SIZE = 12
TEST_SIZE = 12


def xls_to_csv():
    data_ea = pd.read_excel(DATA_PATH_XLS, sheet_name=0, header=0, index_col=0, usecols=[i for i in range(3, 11)])
    data_ea = data_ea[:][1:]
    data_ea.columns = [i.upper() for i in data_ea.columns]
    data_ea.to_csv(DATA_PATH_CSV_EA)

    data_us = pd.read_excel(DATA_PATH_XLS, sheet_name=1, header=0, index_col=0, usecols=[3, 4, 6, 8, 10, 9, 11, 12])
    data_us = data_us[:][1:]
    data_us = data_us[[u'CPI', u'GDP', u'UR', u'IR ', u'IR10', u'LR10 - IR', u' U.S. Dollars to One Euro']]
    data_us.columns = data_ea.columns
    data_us.to_csv(DATA_PATH_CSV_US)


def get_ea_data(drop_na=True):
    data = pd.read_csv(DATA_PATH_CSV_EA, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq='Q')
    if drop_na:
        remove_na(data)
    return data


def get_us_data(drop_na=True):
    data = pd.read_csv(DATA_PATH_CSV_US, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq='Q')
    if drop_na:
        remove_na(data)
    return data


def get_data_dict(drop_na=True):
    data_ea = get_ea_data(drop_na=drop_na)
    data_us = get_us_data(drop_na=drop_na)
    return dict(EA=data_ea, US=data_us)


def get_data_frame(drop_na=True):
    data_ea = get_ea_data(drop_na=drop_na)
    data_us = get_us_data(drop_na=drop_na)

    columns = pd.MultiIndex.from_product([['EA', 'US'], data_ea.columns])

    data_frame = pd.DataFrame(index=data_ea.index, columns=columns)
    data_frame.at[:, ['EA']] = data_ea.values
    data_frame.at[:, ['US']] = data_us.values

    return data_frame


def get_flat_data_frame(drop_na=True):
    data_ea = get_ea_data(drop_na=drop_na)
    data_us = get_us_data(drop_na=drop_na)

    ea_columns = ['EA ' + i for i in data_ea.columns]
    us_columns = ['US ' + i for i in data_us.columns]

    data_frame = pd.DataFrame(index=data_ea.index, columns=ea_columns + us_columns)
    data_frame.at[:, ea_columns] = data_ea.values
    data_frame.at[:, us_columns] = data_us.values

    return data_frame


def train_val_test_split(data, val_size, test_size):
    if isinstance(val_size, float):
        val_size = int(val_size * len(data))

    if isinstance(test_size, float):
        test_size = int(test_size * len(data))

    split = []
    split1 = train_test_split(data, shuffle=False, test_size=val_size + test_size)

    for i in range(len(split1)):
        if i % 2 is 0:
            split.append(split1[i])
        else:
            new_split = train_test_split(split1[i], shuffle=False, test_size=test_size)
            split.extend(new_split)

    return split


def remove_na(data):
    if data.isnull().values.any():
        warnings.warn('Dropped NA values from time series.', stacklevel=2)
    return data.dropna(axis=0, how='any', inplace=True)


def get_xy_data(data, x_lags=1, y_lags=1):
    # Transform X with approapriate lag values
    index = [(i, 'x' + str(j)) for i in data.columns for j in range(x_lags)]
    index = pd.MultiIndex.from_tuples(index, names=['variable', 'lag'])
    x = pd.DataFrame(np.zeros((len(data), len(index))), index=data.index, columns=index, dtype='float64')
    for i in range(x_lags):
        x[[(j, 'x' + str(i)) for j in data.columns]] = data.shift(i + y_lags)

    x = x.tail(-x_lags - y_lags + 1)

    # Transform Y with approapriate lag values
    index = [(i, 'y' + str(j)) for i in data.columns for j in range(y_lags)]
    index = pd.MultiIndex.from_tuples(index, names=['variable', 'lag'])
    y = pd.DataFrame(np.zeros((len(data), len(index))), index=data.index, columns=index, dtype='float64')

    for i in range(y_lags):
        y[[(j, 'y' + str(i)) for j in data.columns]] = data.shift(i)

    y = y.tail(-x_lags - y_lags + 1)

    return x, y


def get_data_formatted(country, var_dict, x_lag, y_lag, val_size, test_size):
    data = get_data_dict(drop_na=True)
    data = data[country]
    X, Y = get_xy_data(data, x_lag)

    x = X.reindex(labels=var_dict['x'], axis='columns', level=0, copy=True)
    y = Y.reindex(labels=var_dict['y'], axis='columns', level=0, copy=True)

    x_train, x_val, x_test = train_val_test_split(x, val_size=val_size, test_size=test_size)
    y_train, y_val, y_test = train_val_test_split(y, val_size=val_size, test_size=test_size)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_data_in_shape(country, vars, x_lags=1, y_lags=1):
    '''
    Formats model input and target data and returns a test and training sets
    :param country: EA or US
    :param vars: ([inputs],[outputs]) tuple of input vars list and output vars list
    :param x_lags: (x lags,y lags) tuple of ints
    :param split: test set size
    :return: x_train, y_train, x_test, y_test
    '''
    data = get_data_dict(drop_na=True)
    data = data[country]
    X, Y = get_xy_data(data, x_lags, y_lags=y_lags)

    x = X.reindex(labels=vars[0], axis='columns', level=0, copy=True)
    y = Y.reindex(labels=vars[1], axis='columns', level=0, copy=True)

    return x, y
