import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error as mse

from src.neuralnets.forecast_model.forecast_models import ForecastModel
from sklearn.model_selection import PredefinedSplit, cross_validate, TimeSeriesSplit
from collections import OrderedDict
from scoop.futures import map
from keras import backend as K



def prediction_scorer(estimator, x, y):
    loss = estimator.score(x, y)
    if isinstance(loss, list):
        return loss[0]
    return loss


def forecast_scorer(estimator, x, y):
    loss = estimator.score_forecast(x, y)
    if isinstance(loss, list):
        return loss[0]
    return loss


class ForecastRegressor(KerasRegressor):

    def predict(self, x, **kwargs):
        kwargs = self.filter_sk_params(ForecastModel.predict, kwargs)
        return np.squeeze(self.model.predict(x, **kwargs))

    def forecast(self, x, y, **kwargs):
        kwargs = self.filter_sk_params(ForecastModel.forecast, kwargs)
        return np.squeeze(self.model.forecast(x, y, **kwargs))

    def score(self, x, y, **kwargs):
        kwargs = self.filter_sk_params(ForecastModel.evaluate, kwargs)
        loss = self.model.evaluate(x, y, **kwargs)
        return loss

    def score_forecast(self, x, y, **kwargs):
        kwargs = self.filter_sk_params(ForecastModel.evaluate_forecast, kwargs)
        loss = self.model.evaluate_forecast(x, y, **kwargs)
        return loss


from src.utils import data_utils
import pandas as pd


class ModelWrapper(ForecastRegressor):
    def __init__(self, build_fn, data_params=None, params=None):
        self.data_set = False
        self.fitted = False

        self.x = None
        self.y = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        if data_params is not None:
            self.set_data_params(**data_params)

        super(ModelWrapper, self).__init__(build_fn, **params)

    def check(self):
        if not self.data_set:
            raise ValueError('Data has not been set!')

    def set_data(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.data_set = True
        self.fitted = False

    def get_data(self):
        self.check()
        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

    def set_data_params(self, **data_params):
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = data_utils.get_data_formatted(
            **data_params)

        self.data_set = True
        self.fitted = False

    def _get_xy(self, set):
        if set == 'train':
            return self.x_train, self.y_train
        elif set == 'val':
            return self.x_val, self.y_val
        elif set == 'test':
            return self.x_test, self.y_test
        else:
            raise ValueError('Set must be train, val or test')

    def _fit(self, x, y):
        self.check()
        self.fitted = True
        return super(ModelWrapper, self).fit(self.x_train, self.y_train)

    def fit(self):
        return self._fit(self.x_train, self.y_train)

    def _predict(self, x):
        self.check()
        if not self.fitted:
            self.fit()

        return super(ModelWrapper, self).predict(x)

    def _forecast(self, x, y):
        self.check()
        if not self.fitted:
            self.fit()

        return super(ModelWrapper, self).forecast(x, y)

    def _evaluate(self, x, y):
        prediction = self._predict(x)
        loss = mse(y, prediction)
        return loss

    def predict(self, set):
        x, _ = self._get_xy(set)
        return self._predict(x)

    def forecast(self, set):
        x, y = self._get_xy(set)
        return self._forecast(x, y)

    def evaluate_prediction(self, set):
        x, y = self._get_xy(set)
        y_pred = self._predict(x)
        loss = mse(y, y_pred, multioutput='raw_values')
        return loss

    def evaluate_forecast(self, set):
        x, y = self._get_xy(set)
        y_pred = self._forecast(x, y)
        loss = mse(y, y_pred, multioutput='raw_values')
        return loss

    def score(self, x, y, **kwargs):
        return self._evaluate(x, y)

    def validate(self, cv_splits, num_runs):
        x = pd.concat([self.x_train, self.x_val], axis=0)
        y = pd.concat([self.y_train, self.y_val], axis=0)

        if cv_splits == 1:
            splitter = PredefinedSplit([-1 for _ in range(len(x) - 12)] + [0 for _ in range(12)])
            split = list(splitter.split(X=x, y=y)) * num_runs
        else:
            splitter = TimeSeriesSplit(cv_splits, max_train_size=len(x) - 12)
            split = list(splitter.split(X=x, y=y)) * num_runs

        res = map(self._validate, split)
        res = np.mean(list(res), axis=0)

        return list(res)

    def _validate(self, splt):

        x = pd.concat([self.x_train, self.x_val], axis=0)
        y = pd.concat([self.y_train, self.y_val], axis=0)

        x_train = x.iloc[splt[0], :]
        x_val = x.iloc[splt[1], :]
        y_train = y.iloc[splt[0], :]
        y_val = y.iloc[splt[1], :]

        self._fit(x_train, y_train)
        tr_res = self._evaluate(x_train, y_train)
        vl_res = self._evaluate(x_val, y_val)
        # print tr_res, vl_res

        return [tr_res, vl_res]

    def fit_eval_pred(self, set):
        self.fit()
        return self.evaluate_prediction(set)

    def fit_eval_fcast(self, set):
        self.fit()
        return self.evaluate_forecast(set)

    def evaluate_all(self, num_runs):
        keys = ['train pred', 'train fcast', 'val pred', 'val fcast', 'test pred', 'test fcast']
        result = OrderedDict([(i, []) for i in keys])
        result['train pred'] = list(map(self.fit_eval_pred, ['train'] * num_runs))
        K.clear_session()
        result['val pred'] = list(map(self.fit_eval_pred, ['val'] * num_runs))
        K.clear_session()
        result['test pred'] = list(map(self.fit_eval_pred, ['test'] * num_runs))
        K.clear_session()
        result['train fcast'] = list(map(self.fit_eval_fcast, ['train'] * num_runs))
        K.clear_session()
        result['val fcast'] = list(map(self.fit_eval_fcast, ['val'] * num_runs))
        K.clear_session()
        result['test fcast'] = list(map(self.fit_eval_fcast, ['test'] * num_runs))
        K.clear_session()

        return result


def model(neurons):
    from keras import layers, activations, losses
    input = layers.Input(shape=(1,))
    layer = layers.Dense(neurons, activation='relu')(input)
    output = layers.Dense(1, activation='relu')(layer)

    model = ForecastModel(inputs=input, outputs=output)
    model.compile('adam', losses.mse)
    return model


def main():
    import src.neuralnets.hypersearch
    params = OrderedDict()
    params['neurons'] = 10
    params['epochs'] = 100

    data_params = OrderedDict()
    data_params['country'] = 'EA'
    data_params['var_dict'] = {'x': ['CPI'], 'y': ['CPI']}
    data_params['x_lag'] = 1
    data_params['y_lag'] = 1
    data_params['val_size'] = 12
    data_params['test_size'] = 12

    wrapper = ModelWrapper(model, data_params, params)
    print wrapper.validate(5, 2)
    # print wrapper.evaluate_all(10)
    # for i in range(10):
    #     print wrapper.evaluate_all(2)


if __name__ == '__main__':
    main()
