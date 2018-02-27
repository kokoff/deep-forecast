import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor, BaseWrapper
from sklearn.metrics import mean_squared_error as mse

from src.neuralnets.forecast_model.forecast_models import ForecastModel
from sklearn.model_selection import PredefinedSplit, cross_validate, TimeSeriesSplit
from collections import OrderedDict
from scoop.futures import map
from keras import backend as K
from matplotlib import pyplot as plt
from src.utils import data_utils
import pandas as pd
import __builtin__


class ForecastRegressor(BaseWrapper):
    def __init__(self, build_fn, data_params=None, params=None):
        self.data_params = None
        self.variables = None

        self.is_data_set = False
        self.is_fitted = False
        self.is_multioutput = None

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

        super(ForecastRegressor, self).__init__(build_fn, **params)

    def set_data_params(self, **data_params):
        self.x, self.y = data_utils.get_data_in_shape(**data_params)
        self.x_train, self.x_val, self.x_test, = data_utils.train_val_test_split(self.x, 12, 12)
        self.y_train, self.y_val, self.y_test = data_utils.train_val_test_split(self.y, 12, 12)

        self.data_params = data_params
        self.is_multioutput = len(data_params['vars'][1]) > 1
        self.variables = data_params['vars'][1]
        self.is_data_set = True
        self.is_fitted = False

    def get_data_params(self):
        return self.data_params

    def check_data_params(self):
        if not self.is_data_set:
            raise ValueError('Data has not been set!')

    def _get_data_fold(self, fold):
        if fold == 'train':
            return self.x_train, self.y_train
        elif fold == 'val':
            return self.x_val, self.y_val
        elif fold == 'test':
            return self.x_test, self.y_test
        else:
            raise ValueError('Set must be train, val or test')

    def _fit(self, x, y, **kwargs):
        self.check_data_params()
        self.is_fitted = True
        return super(ForecastRegressor, self).fit(x, y, **kwargs)

    def _predict(self, x, **kwargs):
        self.check_data_params()
        if not self.is_fitted:
            self.fit()

        kwargs = self.filter_sk_params(ForecastModel.predict, kwargs)
        return np.squeeze(self.model.predict(x, **kwargs))

    def _forecast(self, x, y, **kwargs):
        self.check_data_params()
        if not self.is_fitted:
            self.fit()

        kwargs = self.filter_sk_params(ForecastModel.forecast, kwargs)
        return np.squeeze(self.model.forecast(x, y, **kwargs))

    def _loss_function(self, y, y_pred):
        loss = K.eval(self.model.loss_functions[0](y.T, y_pred))
        if self.is_multioutput:
            sm = np.sum(loss)
            loss = [sm] + list(loss)

        return loss

    def _evaluate_prediction(self, x, y, **kwargs):
        y_pred = self._predict(x, **kwargs)
        loss = self._loss_function(y, y_pred)
        return loss

    def _evaluate_forecast(self, x, y, **kwargs):
        y_pred = self._forecast(x, y, **kwargs)
        loss = self._loss_function(y, y_pred)
        return loss

    def fit(self, **kwargs):
        return self._fit(self.x_train, self.y_train, **kwargs)

    def predict(self, fold, **kwargs):
        x, _ = self._get_data_fold(fold)
        return self._predict(x, **kwargs)

    def forecast(self, fold, **kwargs):
        x, y = self._get_data_fold(fold)
        return self._forecast(x, y, **kwargs)

    def evaluate_prediction(self, fold, refit=False, **kwargs):
        if refit:
            self.fit()

        x, y = self._get_data_fold(fold)
        loss = self._evaluate_prediction(x, y, **kwargs)
        return loss

    def evaluate_forecast(self, fold, refit=False, **kwargs):
        if refit:
            self.fit()

        x, y = self._get_data_fold(fold)
        loss = self._evaluate_forecast(x, y, **kwargs)
        return loss

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

        K.clear_session()

        return res[0][0], res[1][0]

    def _validate(self, split):

        x = self.x
        y = self.y

        x_train = x.iloc[split[0], :]
        x_val = x.iloc[split[1], :]
        y_train = y.iloc[split[0], :]
        y_val = y.iloc[split[1], :]

        self._fit(x_train, y_train)
        tr_res = self._evaluate_prediction(x_train, y_train)
        vl_res = self._evaluate_prediction(x_val, y_val)

        if self.is_multioutput:
            tr_res = tr_res[0]
            vl_res = vl_res[0]

        return [vl_res, tr_res]

    def evaluate_losses(self, num_runs):
        variables = self.variables
        pred_keys = ['train pred', 'val pred', 'test pred']
        fcast_keys = ['train fcast', 'val fcast', 'test fcast']
        sets = ['train', 'val', 'test']

        result = OrderedDict()

        for pred_key, fcast_key, set in zip(pred_keys, fcast_keys, sets):
            pred_res = list(map(self.evaluate_prediction, [set] * num_runs, [True] * num_runs))
            result[pred_key] = np.mean(np.squeeze(pred_res), axis=-1)

            fcast_res = list(map(self.evaluate_forecast, [set] * num_runs, [True] * num_runs))
            result[fcast_key] = np.mean(np.squeeze(fcast_res), axis=-1)

            K.clear_session()

        if self.is_multioutput:
            result = pd.DataFrame(result, index=['total'] + variables)
        else:
            result = pd.DataFrame(result, index=variables)

        return result

    def _get_estimates(self, forecast=False):

        if forecast:
            pred_func = self.predict
            labels = ['train prediction', 'val prediction', 'test prediction']
        else:
            pred_func = self.forecast
            labels = ['train forecast', 'val forecast', 'test forecast']

        num_vars = len(self.variables)
        y = pd.concat([self.y_train, self.y_val, self.y_test], axis=0)
        sets = ['train', 'val', 'test']
        data_series = [self.x_train, self.x_val, self.x_test]
        predictions = []

        self.fit()

        # true y
        temp = pd.DataFrame(y.values, index=y.index, columns=['true values'] * num_vars)
        predictions.append(np.split(temp, num_vars, axis=1))

        for set, series, label in zip(sets, data_series, labels):
            temp = pd.DataFrame(pred_func(set).T, index=series.index,
                                columns=[label] * num_vars)
            predictions.append(np.split(temp, num_vars, axis=1))

        predictions = __builtin__.map(list, zip(*predictions))

        for i in range(len(predictions)):
            predictions[i] = pd.concat(predictions[i], axis=1)

        return predictions

    def get_predictions(self):
        prediction = self._get_estimates()
        return prediction

    def get_forecasts(self):
        forecast = self._get_estimates(forecast=True)
        return forecast

    def get_predictions_and_forecasts(self):
        predictions = self.get_predictions()
        forecasts = self.get_forecasts()
        return predictions, forecasts


def model(neurons):
    from keras import layers, activations, losses
    from forecast_models import create_input_layers, create_output_layers
    inputs, layer = create_input_layers(2, 1)

    layer = layers.Dense(neurons, activation='relu')(layer)

    outputs = create_output_layers(1, 1, layer)

    model = ForecastModel(inputs=inputs, outputs=outputs)
    model.compile('adam', losses.mse)
    return model


def main():
    params = OrderedDict()
    params['neurons'] = 10
    params['epochs'] = 100

    data_params = OrderedDict()
    data_params['country'] = 'EA'
    data_params['vars'] = (['CPI', 'GDP'], ['CPI'])
    data_params['lags'] = (1, 1)

    wrapper = ForecastRegressor(model, data_params, params)
    print wrapper.validate(5, 2)
    print wrapper.evaluate_losses(10)


if __name__ == '__main__':
    main()
