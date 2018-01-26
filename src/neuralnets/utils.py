import numpy as np
import pandas as pd


# Returns x with appropriate lag values and y
def get_xy_data(data, lags=1):
    index = [(i, 'x' + str(j)) for i in data.columns for j in range(lags)]
    index = pd.MultiIndex.from_tuples(index, names=['variable', 'lag'])
    x = pd.DataFrame(np.zeros((len(data), len(index))), index=data.index, columns=index, dtype='float64')

    for i in range(lags):
        x[[(j, 'x' + str(i)) for j in data.columns]] = data.shift(-i)

    x = x.head(-lags)
    y = data.shift(-lags).head(-lags)

    return x, y
