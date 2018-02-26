import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor

from src.neuralnets.forecast_model.forecast_models import ForecastModel


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
