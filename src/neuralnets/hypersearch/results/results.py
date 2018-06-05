import pandas as pd
import os
from visualisation import ResultsPlotter, plot_predictions
from warnings import warn


def get_name_from_data_params(data_params):
    name = ''
    name += data_params['country'] + '_['
    name += 'many' if len(data_params['vars'][0]) > 1 else 'one'
    name += ']_['
    name += 'many' if len(data_params['vars'][1]) > 1 else 'one'
    name += ']'
    # name += str(data_params['lags'])
    return name


def error_prone(func):
    def try_catch(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            warn('Did not manage to save some of the results!', stacklevel=2)
            return func(*args, **kwargs)

    return try_catch


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


class ResultManager:

    def __init__(self, data_params, best_params, log, performance, predictions, forecasts):
        country = data_params['country']
        variables = data_params['vars'][1]
        log = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in log.items()]))
        best_params = pd.DataFrame([best_params])
        self.dir = get_name_from_data_params(data_params)

        if len(variables) == 1:
            self.result = Result(country, variables[0], log, best_params, performance, predictions[0], forecasts[0])
        else:
            self.result = MultiResult(country, variables, log, best_params, performance, predictions, forecasts)

    def __str__(self):
        try:
            return str(self.result)
        except Exception as e:
            print e
            return ''

    def save(self, directory):
        directory = os.path.join(directory, self.dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.result.save(directory)


class Result(object):

    def __init__(self, country, variable, log, best_params, performance, predictions, forecasts):
        self.country = country
        self.variables = variable
        self.name = '_'.join([country, variable])

        self.log = log
        self.params = best_params
        self.performance = performance
        self.predictions = predictions
        self.forecasts = forecasts

    def __str__(self):
        string = ''
        string += '------------------------------------------------------------------------------\n'
        string += 'Result: ' + self.name + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.log) + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.params) + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.performance) + '\n'
        # string += '------------------------------------------------------------------------------\n'
        # string += str(self.predictions) + '\n'
        # string += '------------------------------------------------------------------------------\n'
        # string += str(self.forecasts) + '\n'
        string += '------------------------------------------------------------------------------\n\n'

        return string

    def save(self, directory):
        make_dir(directory)

        var_directory = os.path.join(directory, self.name)
        make_dir(var_directory)

        self.save_log(var_directory)
        # self.save_log_plots(var_directory)

        if self.is_performance_better(var_directory):
            self.save_params(var_directory)
            self.save_performance(var_directory)
            self.save_prediction(var_directory)
            self.save_forecasts(var_directory)
            # self.save_performance_plots(var_directory)

    def is_performance_better(self, directory):
        if os.path.exists(os.path.join(directory, 'performance.csv')):
            old_perf = pd.read_csv(os.path.join(directory, 'performance.csv'))
            return old_perf.reset_index().at[0, 'val pred'] > self.performance.reset_index().at[0, 'val pred']
        else:
            return True

    def save_prediction(self, directory):
        predictions_file = os.path.join(directory, 'prediction.csv')
        self.predictions.to_csv(predictions_file)

    def save_forecasts(self, directory):
        predictions_file = os.path.join(directory, 'forecast.csv')
        self.forecasts.to_csv(predictions_file)

    def save_performance(self, directory):
        params_path = os.path.join(directory, 'performance.csv')
        self.performance.to_csv(params_path, index=False)

    def save_params(self, directory):
        params_path = os.path.join(directory, 'parameters.csv')
        self.params.to_csv(params_path, index=False)

    def save_log(self, directory):
        log_file = os.path.join(directory, 'log.csv')

        if os.path.exists(log_file):
            old_res = pd.read_csv(log_file, index_col=False)

            if set(old_res.columns) != set(self.log.columns):
                print 'Conflicting parameter results!'

            if len(list(self.log)) > len(list(old_res)):
                res = pd.concat([old_res, self.log], join_axes=[self.log.columns])
            else:
                res = pd.concat([old_res, self.log], join_axes=[old_res.columns])
            res.drop_duplicates(res.columns[:-2], inplace=True)
        else:
            res = self.log

        res.to_csv(log_file, index=False)

    def save_log_plots(self, directory):
        figures_directory = os.path.join(directory, 'parameter_figures')
        make_dir(figures_directory)

        log_path = os.path.join(directory, 'log.csv')

        plotter = ResultsPlotter(log_path, figures_directory)
        plotter.plot_all()

    def save_performance_plots(self, directory):
        figures_directory = os.path.join(directory, 'performance_figures')
        make_dir(figures_directory)

        predictions_path = os.path.join(figures_directory, 'prediction.pdf')
        plot_predictions(self.predictions, self.country, self.variables, predictions_path)

        forecast_path = os.path.join(figures_directory, 'forecast.pdf')
        plot_predictions(self.forecasts, self.country, self.variables, forecast_path)


class MultiResult:

    def __init__(self, country, variable, log, best_params, performance, predictions, forecasts):
        self.results = [None for _ in range(len(variable))]

        for i in range(len(variable)):
            self.results[i] = Result(country, variable[i], log, best_params, performance.iloc[[i + 1]], predictions[i],
                                     forecasts[i])

    def __str__(self):
        string = ''
        for result in self.results:
            string += str(result)
        return string

    def save(self, directory):
        for result in self.results:
            result.save(directory)
