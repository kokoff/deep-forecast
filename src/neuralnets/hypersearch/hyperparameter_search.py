import os
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import ParameterGrid

from optimizers import RSOptimizer, PSOptimizer, GSOptimizer
from results import ResultManager
from src.neuralnets.forecast_model.forecast_model_wrapper import ForecastRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Logger:
    def __init__(self):
        self._log = OrderedDict()
        self.best_params = None
        self.best_results = {'train': np.inf, 'val': np.inf}
        self.log_initialised = False

    def get_log(self):
        return self._log

    def init_log(self, params):
        for key in params:
            if isinstance(params[key], list):
                for i, val in enumerate(params[key]):
                    self._log[key + str(i)] = []
            else:
                self._log[key] = []
        self._log.update({'train': [], 'val': []})
        self.log_initialised = True

    def log(self, params, val, train):

        if not self.log_initialised:
            self.init_log(params)

        self._log['val'].append(val)
        self._log['train'].append(train)

        for param, value in params.iteritems():
            if isinstance(value, list):
                for i, val in enumerate(value):
                    if param + str(i) not in self._log:
                        self._log[param + str(i)] = []
                    self._log[param + str(i)].append(val)
            else:
                self._log[param].append(value)

        if val < self.best_results['val']:
            self.best_params = params
            self.best_results['val'] = val
            self.best_results['train'] = train

    def get_best_params(self):
        return self.best_params

    def get_best_results(self):
        return self.best_results


class Runner:

    def __init__(self, validator, cv_splits, runs):
        self.logger = Logger()
        self.validator = validator
        self.cv_splits = cv_splits
        self.runs = runs

    def run(self, **params):
        print params,

        self.validator.set_params(**params)
        val, train = self.validator.validate(self.cv_splits, self.runs)

        print '\tval', val, 'train', train

        self.logger.log(params, val, train)

        return val

    def get_log(self):
        return self.logger.get_log()

    def get_best_params(self):
        return self.logger.get_best_params()

    def get_best_results(self):
        return self.logger.get_best_results()


def experiment_dir(data_param):
    basedir = 'experiments'

    if not os.path.exists(basedir):
        os.mkdir(basedir)

    if len(data_param['var_dict']['x']) == 1:
        path = 'one_to_one'
    elif len(data_param['var_dict']['x']) == 7 and len(data_param['var_dict']['y']) == 1:
        path = 'many_to_one'
    elif len(data_param['var_dict']['x']) == 7 and len(data_param['var_dict']['x']) == 7:
        path = 'many_to_many'
    else:
        i = 0
        path = 'experiment'
        while os.path.exists(os.path.join(basedir, path + str(i))):
            i += 1
        path = os.path.join(path + str(i))

    full_path = os.path.join(basedir, path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    return full_path


class HyperSearch:
    def __init__(self, solver, cv_splits=5, validation_runs=2, eval_runs=10, output_dir='experiments', **solver_kwargs):
        if solver == 'pso':
            self.solver = PSOptimizer(**solver_kwargs)
        elif solver == 'gso':
            self.solver = GSOptimizer()
        elif solver == 'rso':
            self.solver = RSOptimizer(**solver_kwargs)
        else:
            raise ValueError('Solver must be one of pso, gso, rso!')

        self.runs = validation_runs
        self.cv_splits = cv_splits
        self.output_dir = output_dir
        self.eval_runs = eval_runs

    def hyper_data_search(self, build_fn, data_params_dict, params):
        data_params = ParameterGrid(data_params_dict)

        for data_param in data_params:
            print 'data params:\t', data_param

            result = self.hyper_search(build_fn, data_param, params)
            print result

    def hyper_search(self, build_fn, data_param, params):

        model = ForecastRegressor(build_fn, data_param, params)
        runner = Runner(model, self.cv_splits, self.runs)

        res = self.solver.optimize(runner.run, params)
        print 'best params:\t', res.params
        print 'best score:\t', res.score
        print 'run time:\t', res.time

        # evaluate model
        model.set_params(**res.params)
        performance = model.evaluate_losses(self.eval_runs)
        predictions = model.get_predictions()
        forecasts = model.get_forecasts()

        # create results
        result = ResultManager(data_param, res.params, runner.get_log(), performance, predictions,
                               forecasts)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        result.save(self.output_dir)
        return result
