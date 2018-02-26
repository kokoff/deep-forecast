import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from io import StringIO
from rpy2 import robjects

r = robjects.r

sns.set()

arima_dir = '/home/skokov/project/experiments/arima'


def read_predictions(file_path):
    with open(file_path, 'r') as f:
        csv = f.read()
        csv = csv.replace(' Q', 'Q')

    df = pd.read_csv(StringIO(unicode(csv)), sep=',', index_col=0, parse_dates=True, infer_datetime_format=True)
    df.index = df.index.to_period()

    return df


def plot_predictions(df, label):
    df.plot()
    plt.xlabel('Time')
    plt.ylabel(label)


def plot_results():
    for dir in os.listdir(arima_dir):
        res_dir = os.path.join(arima_dir, dir)

        predictions_file = os.path.join(res_dir, 'predictions.csv')
        forecasts_file = os.path.join(res_dir, 'forecasts.csv')

        prediction_fig_path = os.path.join(res_dir, 'prediction.pdf')
        forecast_fig_path = os.path.join(res_dir, 'forecast.pdf')

        label = ' '.join(dir.split('_'))

        predictions = read_predictions(predictions_file)
        forecasts = read_predictions(forecasts_file)

        plot_predictions(predictions, label)
        plt.savefig(prediction_fig_path)
        plt.show()

        plot_predictions(forecasts, label)
        plt.savefig(forecast_fig_path)
        plt.show()


def main():
    robjects.globalenv["output_dir"] = '/home/skokov/project/experiments/arima'
    robjects.r.source("arima.R")
    plot_results()


if __name__ == '__main__':
    main()
