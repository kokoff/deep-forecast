import os
import sys
from collections import OrderedDict

import numpy as np
import optunity
from keras import backend as K
from optunity import functions, search_spaces
from optunity.constraints import wrap_constraints
from sklearn.model_selection import ParameterGrid

from optimizers import RSOptimizer, PSOptimizer, GSOptimizer
from src.utils import data_utils
from validation import ModelValidator, ModelEvaluator
from results import ResultManager
from src.neuralnets.forecast_model.forecast_model_wrapper import ModelWrapper
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
            self._log[key] = []
        self._log.update({'train': [], 'val': []})
        self.log_initialised = True

    def log(self, params, val, train):

        if not self.log_initialised:
            self.init_log(params)

        self._log['val'].append(val)
        self._log['train'].append(train)

        for param, value in params.iteritems():
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

    def __init__(self, validator):
        self.logger = Logger()
        self.validator = validator

    def run(self, **params):
        K.clear_session()

        print params,

        self.validator.set_params(**params)
        val, train = self.validator.validate(5,2)

        print 'val', val, 'train', train

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


def get_name_from_data_params(data_params):
    name = ''
    name += data_params['country'] + '_['
    name += '_'.join(data_params['vars'][0]) + ']_['
    name += '_'.join(data_params['vars'][1]) + ']_'
    name += str(data_params['lags'][0]) + '_'
    name += str(data_params['lags'][1])
    return name


class HyperSearch:
    def __init__(self, solver, cv_splits, validation_runs, eval_runs, output_dir='experiments', **solver_kwargs):
        '''Perform hyper parameter search using optunity as backend. Possible solvers include
        grid search     args: None
        random search   args: num_evals
        particle swarm  args: num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0
        sobol           args: num_evals, seed=None, skip=None
        '''
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

        # x, y = data_utils.get_data_in_shape(**data_param)

        validator = ModelWrapper(build_fn, data_param, params)
        runner = Runner(validator)

        res = self.solver.optimize(runner.run, params)
        print 'best params:\t', res.params
        print 'best score:\t', res.score
        print 'run time:\t', res.time

        # evaluator = ModelEvaluator(build_fn, x, y, data_param)
        # performance = evaluator.evaluate(self.eval_runs, **res.params)
        # predictions, forecasts = evaluator.predict(**res.params)

        print validator.evaluate_all(10)

        # result = ResultManager(data_param, res.params, runner.get_log(), performance, predictions, forecasts)
        #
        # print result
        #
        # if not os.path.exists(self.output_dir):
        #     os.mkdir(self.output_dir)
        # out_dir = os.path.join(self.output_dir, get_name_from_data_params(data_param))
        # result.save(out_dir)
        # return result
