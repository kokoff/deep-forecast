import pandas as pd
import os
import warnings

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data'))
DATA_PATH_XLS = os.path.join(DATA_PATH, 'Data_ILP.xls')
DATA_PATH_CSV_EA = os.path.join(DATA_PATH, 'EA.csv')
DATA_PATH_CSV_US = os.path.join(DATA_PATH, 'US.csv')


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


def get_ea_data():
    data = pd.read_csv(DATA_PATH_CSV_EA, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq='Q')
    return data


def get_us_data():
    data = pd.read_csv(DATA_PATH_CSV_US, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq='Q')
    return data


def get_data():
    data_ea = get_ea_data()
    data_us = get_us_data()
    return dict(EA=data_ea, US=data_us)


def remove_na(series):
    if series.hasnans:
        warnings.warn('Dropped NA values from time series.',stacklevel=2)
        return series.dropna(inplace=True)
