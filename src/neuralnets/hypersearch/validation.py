from collections import OrderedDict

import numpy as np
from sklearn.model_selection import PredefinedSplit, TimeSeriesSplit, cross_validate, train_test_split

from src.neuralnets.forecast_model.forecast_model_wrapper import prediction_scorer, ForecastRegressor
from src.utils.data_utils import VAL_SIZE, TEST_SIZE, train_val_test_split
import pandas as pd
from keras import backend as K


class Base(object):
    def __init__(self, build_fn, x, y):
        self.estimator = ForecastRegressor(build_fn)
        self.x = x
        self.y = y
        self.sizes = OrderedDict()
        self.sizes['num_inputs'] = len(x.columns.levels[0])
        self.sizes['input_size'] = len(x.columns.levels[1])
        self.sizes['num_outputs'] = len(y.columns.levels[0])
        self.sizes['output_size'] = len(y.columns.levels[1])

    def _set_params(self, **params):
        params.update(self.sizes)
        self.estimator.set_params(**params)


class ModelValidator(Base):

    def __init__(self, build_fn, x, y, cv_splits, cv_runs):
        self.cv_splits = cv_splits
        self.cv_runs = cv_runs
        super(ModelValidator, self).__init__(build_fn, x, y)

    def validate(self, **params):
        self._set_params(**params)

        x, _ = train_test_split(self.x, test_size=TEST_SIZE, shuffle=False)
        y, _ = train_test_split(self.y, test_size=TEST_SIZE, shuffle=False)

        if self.cv_splits == 1:
            splitter = PredefinedSplit([-1 for _ in range(len(x) - VAL_SIZE)] + [0 for _ in range(VAL_SIZE)])
            split = list(splitter.split(X=x, y=y)) * self.cv_runs
        else:
            splitter = TimeSeriesSplit(self.cv_splits, max_train_size=len(x) - VAL_SIZE)
            split = list(splitter.split(X=x, y=y)) * self.cv_runs

        scores = cross_validate(self.estimator, x, y, cv=split, n_jobs=4,
                                return_train_score=True, scoring=prediction_scorer)

        return np.mean(scores['test_score']), np.mean(scores['train_score'])


class ModelEvaluator(Base):

    def __init__(self, build_fn, x, y, data_params):
        self.variables = data_params['vars'][1]
        super(ModelEvaluator, self).__init__(build_fn, x, y)

    def fit_and_score(self, x_train, y_train, x_test, y_test):
        print '*',

        self.estimator.fit(x_train, y_train)
        score = self.estimator.score(x_test, y_test)
        return score

    def fit_and_score_forecast(self, x_train, y_train, x_test, y_test):
        print '*',
        self.estimator.fit(x_train, y_train)
        score = self.estimator.score_forecast(x_test, y_test)
        return score

    def evaluate(self, num_runs, **params):
        self._set_params(**params)

        x_train, x_val, x_test = train_val_test_split(self.x, VAL_SIZE, TEST_SIZE)
        y_train, y_val, y_test = train_val_test_split(self.y, VAL_SIZE, TEST_SIZE)

        train = self._eval(x_train, y_train, x_train, y_train, num_runs)
        val = self._eval(x_train, y_train, x_val, y_val, num_runs)
        test = self._eval(x_train, y_train, x_test, y_test, num_runs)
        vars = np.concatenate([train, val, test])

        columns = ['train pred', 'train fcast', 'val pred', 'val fcast', 'test pred', 'test fcast']
        if len(self.variables) > 1:
            index = ['Total'] + self.variables
            evaluations = pd.DataFrame(vars.T, columns=columns, index=index)
        else:
            index = self.variables
            evaluations = pd.DataFrame(vars.T, columns=columns, index=index)

        return evaluations

    def _eval(self, x_train, y_train, x_test, y_test, num_runs):

        print 'eval pred ({0})'.format(num_runs),
        pred = map(lambda _: self.fit_and_score(x_train, y_train, x_test, y_test), list(range(num_runs)))
        pred = np.mean(pred, axis=0)
        K.clear_session()

        print 'eval fcast ({0})'.format(num_runs),
        fcast = map(lambda _: self.fit_and_score_forecast(x_train, y_train, x_test, y_test), list(range(num_runs)))
        fcast = np.mean(fcast, axis=0)
        K.clear_session()

        if len(self.variables) > 1:
            return np.squeeze(np.stack([pred, fcast]))
        else:
            pred = [pred]
            fcast = [fcast]
            return np.stack([pred, fcast])

    def predict(self, **params):
        params.update(self.sizes)
        self.estimator.set_params(**params)

        x_train, x_val, x_test = train_val_test_split(self.x, VAL_SIZE, TEST_SIZE)
        y_train, y_val, y_test = train_val_test_split(self.y, VAL_SIZE, TEST_SIZE)

        self.estimator.fit(x_train, y_train)

        num_vars = len(self.variables)
        predictions = []
        forecasts = []

        # true y
        temp = pd.DataFrame(self.y.values, index=self.y.index, columns=['true values'] * num_vars)
        predictions.append(np.split(temp, num_vars, axis=1))
        forecasts.append(np.split(temp, num_vars, axis=1))

        # predictions
        temp = pd.DataFrame(self.estimator.predict(x_train).T, index=y_train.index,
                            columns=['train prediction'] * num_vars)
        predictions.append(np.split(temp, num_vars, axis=1))

        temp = pd.DataFrame(self.estimator.predict(x_val).T, index=y_val.index, columns=['val prediction'] * num_vars)
        predictions.append(np.split(temp, num_vars, axis=1))

        temp = pd.DataFrame(self.estimator.predict(x_test).T, index=y_test.index,
                            columns=['test prediction'] * num_vars)
        predictions.append(np.split(temp, num_vars, axis=1))

        predictions = map(list, zip(*predictions))

        for i in range(len(predictions)):
            predictions[i] = pd.concat(predictions[i], axis=1)

        # forecasts
        temp = pd.DataFrame(self.estimator.forecast(x_train, y_train).T, index=x_train.index,
                            columns=['train forecast'] * num_vars)
        forecasts.append(np.split(temp, num_vars, axis=1))

        temp = pd.DataFrame(self.estimator.forecast(x_val, y_val).T, index=x_val.index,
                            columns=['val forecast'] * num_vars)
        forecasts.append(np.split(temp, num_vars, axis=1))

        temp = pd.DataFrame(self.estimator.forecast(x_test, y_test).T, index=x_test.index,
                            columns=['test forecast'] * num_vars)
        forecasts.append(np.split(temp, num_vars, axis=1))

        forecasts = map(list, zip(*forecasts))
        for i in range(len(forecasts)):
            forecasts[i] = pd.concat(forecasts[i], axis=1)

        if len(self.variables) == 1:
            predictions = predictions[0]
            forecasts = forecasts[0]

        return predictions, forecasts


def main():
    from src.utils import data_utils
    from src.neuralnets.temp1 import makeModel
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()

    data_param = OrderedDict()
    data_param['country'] = 'EA'
    data_param['vars'] = (['CPI', 'GDP'], ['CPI'])
    data_param['lags'] = (1, 1)

    params = OrderedDict()
    params['neurons'] = 2
    params['epochs'] = 10
    params['batch_size'] = 10
    params['optimiser'] = 'adam'

    x, y = data_utils.get_data_in_shape(**data_param)

    tester = ModelEvaluator(makeModel, x, y, data_param)
    print tester.evaluate(2, **params)
    res = tester.predict(**params)
    # print res

    for i in res:
        for j in i:
            j.plot()
            plt.show()


if __name__ == '__main__':
    main()
