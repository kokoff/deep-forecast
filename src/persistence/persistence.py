import os
from collections import OrderedDict
from itertools import product

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from src.utils import EXPERIMENTS_DIR
from src.utils import data_utils

PERSISTENCE_DIR = os.path.join(EXPERIMENTS_DIR, 'persistence')

sns.set()


def persistence_mse(series):
    columns = ['train pred', 'val pred', 'test pred']
    lossess = OrderedDict([(i, 0) for i in columns])

    y_pred, y_true = data_utils.get_xy_data(series, 1)
    true_data = data_utils.train_val_test_split(y_true, 12, 12)
    pred_data = data_utils.train_val_test_split(y_pred, 12, 12)

    for pred, true, col in zip(pred_data, true_data, columns):
        lossess[col] = mse(true, pred)

    return pd.DataFrame(lossess, index=[0])


def persistence_prediction(series):
    y_pred, y_true = data_utils.get_xy_data(series, 1)
    y_train, y_val, y_test = data_utils.train_val_test_split(y_pred, 12, 12)
    predictions = pd.concat([y_true, y_train, y_val, y_test], axis=1)
    predictions.columns = ['true values', 'train prediction', 'val prediction', 'test prediction']
    return predictions


def plot_predictions(predictions, label, output_dir=None):
    predictions.plot()
    plt.xlabel('Time')
    plt.ylabel(label)
    if output_dir is not None:
        fig_path = os.path.join(output_dir, 'prediction.pdf')
        plt.savefig(fig_path)
        plt.close('all')
    else:
        plt.show()


def evaluate_persistance(series, country, variable, output_dir):
    dir_name = '_'.join([country, variable])
    label = ' '.join([country, variable])
    dir_path = os.path.join(output_dir, dir_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    pred_path = os.path.join(dir_path, 'prediction.csv')
    performance_path = os.path.join(dir_path, 'performance.csv')

    pred = persistence_prediction(series)
    loss = persistence_mse(series)

    pred.to_csv(pred_path)
    loss.to_csv(performance_path)
    plot_predictions(pred, label, output_dir=dir_path)


def main():
    if not os.path.exists(PERSISTENCE_DIR):
        os.mkdir(PERSISTENCE_DIR)

    output_dir = PERSISTENCE_DIR

    for country, variable in product(data_utils.COUNTRIES, data_utils.VARIABLES):
        series = data_utils.get_data_dict()[country][[variable]]
        evaluate_persistance(series, country, variable, output_dir)


if __name__ == '__main__':
    main()
