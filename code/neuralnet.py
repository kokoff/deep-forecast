import numpy as np
import pandas as pd
from pandas import DataFrame
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics
from sklearn.preprocessing import *
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.utils import plot_model, print_summary


def plotMetrics(hist, model):
    plt.figure()

    metrics = model.metrics_names
    num_metrics = len(metrics)
    rows = min(3, num_metrics)
    cols = 1 if num_metrics <= 3 else 2
    print 'Row', rows, 'Col', cols
    # plot metrics
    for i in range(num_metrics):
        ax = plt.subplot(rows, cols, i + 1, title=metrics[i])
        ax.plot(hist.history[metrics[i]], label=metrics[i])
        if 'val_' + metrics[i] in hist.history.keys():
            ax.plot(hist.history['val_' + metrics[i]], label='val_' + metrics[i])
        ax.legend()


def plotPredictions(target, prediction):
    plt.figure()

    ax = plt.subplot(1, 1, 1, title='Target vs Prediction')
    ax.plot(target, 'b-', label='target')
    ax.plot(prediction, 'r--', label='prediction')
    ax.legend()

    # ax = plt.subplot(1, 2, 2, title = 'Target minus prediction')
    # ax.plot(target - prediction, 'p-', label='target - prediction')
    # ax.legend()


def forecast(model, X):
    predictions = X
    res = np.array([X[0]])
    predictions[0] = res
    # print predictions
    print np.array([res])
    print model.predict(res)
    print X[0]
    print X[1]
    print 'Loop'
    for i in range(len(predictions)):
        temp = model.predict(res)
        res = np.array([np.append(res[0, 1:], temp)])
        predictions[i] = temp
    plt.plot(predictions)
    plt.show()


def getData(data_frame, name, lag=0):
    data = data_frame[name]
    N = data_frame.shape[0]
    m = lag + 1

    # Calculate lags
    XY = np.ndarray((N, m + 1))
    for i in range(m + 1):
        XY[:, i] = data.shift(-i)

    # Remove NAs
    na_mask = np.isnan(XY).any(axis=1)
    XY = XY[~na_mask]

    print XY

    # Create X and Y
    X = XY[:, :m]
    Y = XY[:, m:]

    return X, Y


def getModel(input_dim=1, neurons=1):
    # Create Model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='MSE',
                  optimizer='adam',
                  metrics=[])

    return model


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lags", help="display a square of a given number",
                        type=int)
    parser.add_argument("--VAR", help="display a square of a given number",
                        type=str)
    args = parser.parse_args()
    print args.VAR, args.lags

    # read data and drop missing values
    data_file = os.path.join('..', 'data', 'ea.csv')
    data = pd.read_csv(data_file)

    VAR = args.VAR
    INPUT_DIM = args.lags

    X, Y = getData(data, VAR, INPUT_DIM - 1)

    model = KerasRegressor(getModel)
    param_grid = dict(input_dim=[INPUT_DIM],
                      neurons=[5, 10, 15, 20],
                      epochs=[10, 15, 20],
                      shuffle=[False],
                      batch_size=[1, 2, 4],
                      verbose = [False])

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X, Y)
    dfresults = DataFrame.from_dict(grid_result.cv_results_)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    # Record the results
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'experiments')

    # Output table of results
    columns = ['rank_test_score', 'mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']
    columns += [col for col in dfresults.columns if col.startswith('param_')]
    file_path = os.path.join(output_path, VAR + '.csv')
    dfresults.to_csv(file_path, float_format='%.3f', columns=columns, mode='a')

    # Save model config
    best_model = grid_result.best_estimator_.model
    file_path = os.path.join(output_path, VAR + str(INPUT_DIM) + '.json')
    with open(file_path, 'w') as f:
        f.write(best_model.to_json())

    # Output plot
    file_path = os.path.join(output_path, VAR + str(INPUT_DIM) + '.png')
    plotPredictions(Y, best_model.predict(X))
    plt.savefig(file_path, bbox_inches='tight')


if __name__ == '__main__':
    main()
