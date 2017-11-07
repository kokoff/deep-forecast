import numpy as np
import pandas as pd
from pandas import DataFrame
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics
from sklearn.preprocessing import StandardScaler


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

    # Create X and Y
    X = XY[:, :m]
    Y = XY[:, m:]

    return X, Y


def main():
    # read data and drop missing values
    data_file = os.path.join('..', 'data', 'ea.csv')
    data = pd.read_csv(data_file)

    X, Y = getData(data, 'CPI', 0)
    input_dim = X.shape[1]

    # split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)

    # scaler = StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    model = Sequential()

    model.add(Dense(8, activation='relu', input_dim=input_dim))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='MSE',
                  optimizer='adam',
                  metrics=[])

    history = model.fit(x_train, y_train, epochs=500, batch_size=2, validation_split=0.2)
    score = model.evaluate(x_test, y_test)
    prediction = model.predict(X)

    plotMetrics(history, model)
    plotPredictions(y_test, model.predict(x_test))
    #
    plt.show()

    forecast(model, X)



if __name__ == '__main__':
    main()
