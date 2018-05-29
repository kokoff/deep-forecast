import os
import json

import numpy as np
import pandas as pd
from src.utils.data_utils import VARIABLES, COUNTRIES
dir_names = ['_'.join([i, j]) for i in COUNTRIES for j in VARIABLES]


config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config.json')
with open(config_file, 'r') as f:
    config = json.load(f)

EXPERIMENTS_DIR = config['EXPERIMENTS_DIR']

if not os.path.isabs(EXPERIMENTS_DIR):
    EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(config_file), EXPERIMENTS_DIR))

if not os.path.exists(EXPERIMENTS_DIR):
    os.mkdir(EXPERIMENTS_DIR)

print 'EXPERIMENTS_DIR = ', EXPERIMENTS_DIR


def check_experiments_dict(experiments_dict):
    dirs = []
    for i in experiments_dict.values():
        dirs.extend(i)

    assert len(dirs) == len(set(dirs))

    for i in dirs:
        if not os.path.exists(i):
            print i
        assert os.path.exists(i)


def mape(df):
    y_true = df.iloc[:, 0].as_matrix()
    y_pred = df.iloc[:, 1].as_matrix()

    if 0.0 in y_true or 0.0 in y_pred:
        y_pred = y_pred[[y_true != 0]]
        y_true = y_true[[y_true != 0]]

    # y_true[y_true == 0] = 0.0000000000000000000000000000000000000000001
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_errors(path):
    predictions = os.path.join(path, 'prediction.csv')
    forecasts = os.path.join(path, 'forecast.csv')

    prediction = pd.read_csv(predictions, index_col=0)
    train_pred = prediction.iloc[:, [0, 1]].dropna()
    val_pred = prediction.iloc[:, [0, 2]].dropna()
    test_pred = prediction.iloc[:, [0, 3]].dropna()

    keys = ['train pred', 'train fcast', 'val pred', 'val fcast', 'test pred', 'test fcast']
    data = [mape(train_pred), None, mape(val_pred), None, mape(test_pred), None]
    dict = {i: j for i, j in zip(keys, data)}

    res = pd.DataFrame(dict, index=[0], columns=keys)

    if os.path.exists(forecasts):
        forecast = pd.read_csv(forecasts, index_col=0)
        train_fcast = forecast.iloc[:, [0, 1]].dropna()
        val_fcast = forecast.iloc[:, [0, 2]].dropna()
        test_fcast = forecast.iloc[:, [0, 3]].dropna()

        keys = ['train pred', 'train fcast', 'val pred', 'val fcast', 'test pred', 'test fcast']
        data = [mape(train_pred), mape(train_fcast), mape(val_pred), mape(val_fcast), mape(test_pred), mape(test_fcast)]
        dict = {i: j for i, j in zip(keys, data)}

        res = pd.DataFrame(dict, index=[0], columns=keys)

    return res


def calculate_mades():
    for root, dirs, files in os.walk('.'):
        if 'prediction.csv' in files:
            df = get_errors(root)
            path = os.path.join(root, 'performance_made.csv')
            df.to_csv(path)


def get_table(experiments_dict):
    pred_table = pd.DataFrame(0.0, index=dir_names,
                              columns=experiments_dict.keys())
    fcast_table = pred_table.copy()
    val_pred_table = pred_table.copy()
    val_fcast_table = pred_table.copy()

    check_experiments_dict(experiments_dict)
    for key, value in experiments_dict.iteritems():
        for dir in value:
            for root, dirs, files in os.walk(dir):
                if 'prediction.csv' in files:
                    df = get_errors(root)
                    col = key
                    row = os.path.basename(root)

                    pred_table.loc[row, col] = df.loc[0, 'test pred']
                    fcast_table.loc[row, col] = df.loc[0, 'test fcast']
                    val_pred_table.loc[row, col] = df.loc[0, 'val pred']
                    val_fcast_table.loc[row, col] = df.loc[0, 'val fcast']

    pred_table.to_csv('predictions.csv')
    fcast_table.to_csv('forecasts.csv')
    val_pred_table.to_csv('validation_predictions.csv')
    val_fcast_table.to_csv('validation_forecasts.csv')