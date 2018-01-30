import numpy as np
import pandas as pd


# Returns x with appropriate lag values and y
def get_xy_data(data, lags1=1, lags2=1):
    # Transform X with approapriate lag values
    index = [(i, 'x' + str(j)) for i in data.columns for j in range(lags1)]
    index = pd.MultiIndex.from_tuples(index, names=['variable', 'lag'])
    x = pd.DataFrame(np.zeros((len(data), len(index))), index=data.index, columns=index, dtype='float64')
    for i in range(lags1):
        x[[(j, 'x' + str(i)) for j in data.columns]] = data.shift(i + lags2)

    x = x.tail(-lags1 - lags2 + 1)

    # Transform Y with approapriate lag values
    index = [(i, 'y' + str(j)) for i in data.columns for j in range(lags2)]
    index = pd.MultiIndex.from_tuples(index, names=['variable', 'lag'])
    y = pd.DataFrame(np.zeros((len(data), len(index))), index=data.index, columns=index, dtype='float64')

    for i in range(lags2):
        y[[(j, 'y' + str(i)) for j in data.columns]] = data.shift(i)

    y = y.tail(-lags1 - lags2 + 1)

    return x, y


if __name__ == '__main__':
    from src.utils import data_utils

    data = data_utils.get_ea_data()[['CPI']]
    X, Y = get_xy_data(data, 1, 2)

    print data.head(), X.head(), Y.head()
    print data.tail(), X.tail(), Y.tail()
