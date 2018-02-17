import os
from collections import OrderedDict

import numpy as np
import optunity
import pandas as pd
from keras import backend as K
from optunity.solvers import GridSearch, RandomSearch, ParticleSwarm, Sobol
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit

from src.neuralnets.forecast_models import ModelWrapper

from src.neuralnets.hypersearch.results import MultiResult, SingleResult
from sklearn.model_selection import cross_validate
from src.neuralnets.model_selection.model_selection import ForecastRegressor
from validation import ModelEvaluator
from src.utils import data_utils
from optunity import functions, search_spaces
from optunity.constraints import wrap_constraints
import sys
from multiprocessing import Pool
from optunity.parallel import create_pmap
from optimizers import PSOptimizer

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

    def __init__(self, build_fn, x, y, cv_splits, num_runs, all_params):
        self.logger = Logger()
        self.validator = ModelEvaluator(build_fn, x, y, cv_splits, num_runs)
        self.param_manager = ParameterManager(all_params)

    def run(self, **params):
        K.clear_session()
        params = self.param_manager.transform(params)

        print params,

        val, train = self.validator.evaluate(**params)

        print 'val', val, 'train', train

        self.logger.log(params, val, train)

        return val

    def get_log(self):
        return self.logger.get_log()

    def get_best_params(self):
        return self.logger.get_best_params()

    def get_best_results(self):
        return self.logger.get_best_results()

    def get_params(self):
        return self.param_manager.get_ranges()


def get_solver(solver, solver_params, params):
    if solver == 'grid search':
        print 'solver is {0}'.format(solver)
        solver = GridSearch(**params)

    elif solver == 'random search':
        num_evals = solver_params.pop('num_evals', 100)
        print 'solver is {0} with num evals={1}'.format(solver, num_evals)
        solver = RandomSearch(num_evals=num_evals, **params)

    elif solver == 'sobol':
        num_evals = solver_params.pop('num_evals', 100)
        seed = solver_params.pop('seed', None)
        skip = solver_params.pop('skip', None)
        print 'solver is {0} with num_evals={1}, seed={2}, skip={3}'.format(solver, num_evals, seed, skip)
        solver = Sobol(num_evals, seed, skip, **params)

    elif solver == 'particle swarm':
        num_particles = solver_params.pop('num_particles', 10)
        num_generations = solver_params.pop('num_generations', 10)
        max_speed = solver_params.pop('max_speed', None)
        phi1 = solver_params.pop('phi1', 1.5)
        phi2 = solver_params.pop('phi2', 2.0)
        print 'solver is {0} with num_particles={1}, num_generations={2}, max_spped={3}, phi1={4}, phi2={5}'.format(
            solver, num_particles, num_generations, max_speed, phi1, phi2)
        solver = ParticleSwarm(num_particles, num_generations, max_speed, phi1, phi2, **params)
    else:
        raise ValueError(solver, 'is not a valid solver.')
    return solver


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


class ParameterManager:
    def __init__(self, param_dict):
        self.type_dict = self._recurse_types(param_dict)
        self.range_dict = self._recurse_ranges(param_dict)
        self.keys = self._recurse_keys(param_dict)

    def get_ranges(self):
        return self.range_dict

    def get_transforms(self):
        return self.type_dict

    def get_keys(self):
        return self.keys

    def transform(self, params):
        transformed_params = OrderedDict()
        for key in self.keys:
            if key in params:
                if key in self.type_dict and params[key] is not None:
                    transformed_params[key] = self.type_dict[key](params[key])
                else:
                    transformed_params[key] = params[key]

        return transformed_params

    def _recurse_types(self, var):
        if not isinstance(var, dict):
            if isinstance(var, tuple):
                return var[0]
            else:
                return None

        else:
            type_dict = OrderedDict()
            for key, value in var.iteritems():
                res = self._recurse_types(value)

                if res is not None and not isinstance(res, dict):
                    type_dict[key] = res
                elif res is not None:
                    type_dict.update(res)

            return type_dict

    def _recurse_ranges(self, var):
        if not isinstance(var, dict):
            if var is not None:
                return var[1]
            else:
                return None

        else:
            range_dict = dict()
            for key, value in var.iteritems():
                range_dict[key] = self._recurse_ranges(value)

            return range_dict

    def _recurse_keys(self, var):
        if not isinstance(var, dict):
            return []

        else:
            keys = var.keys()
            for key, value in var.iteritems():
                res = self._recurse_keys(value)
                keys.extend(res)
            return keys


class HyperSearch:
    def __init__(self, solver, cv_splits, validation_runs, **kwargs):
        '''Perform hyper parameter search using optunity as backend. Possible solvers include
        grid search     args: None
        random search   args: num_evals
        particle swarm  args: num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0
        sobol           args: num_evals, seed=None, skip=None
        '''
        self.solver_name = solver
        self.solver_params = kwargs
        self.runs = validation_runs
        self.cv_splits = cv_splits

    def run_solver(self, f, params):
        if self.solver_name == 'grid search':
            solver = get_solver(self.solver_name, self.solver_params, params)
            res, detail = optunity.optimize(solver, f, maximize=False, pmap=map)
        else:
            tree = search_spaces.SearchTree(params)
            box = tree.to_box()

            # we need to position the call log here
            # because the function signature used later on is internal logic
            f = functions.logged(f)

            # wrap the decoder and constraints for the internal search space representation
            f = tree.wrap_decoder(f)
            f = wrap_constraints(f, -sys.float_info.max, range_oo=box)

            solver = get_solver(self.solver_name, self.solver_params, box)
            res, detail = optunity.optimize(solver, f, maximize=False, pmap=map, max_evals=0)

        return res, detail

    def hyper_data_search(self, build_fn, data_params_dict, params):
        data_params = ParameterGrid(data_params_dict)

        for data_param in data_params:
            print data_param
            try:
                result = self.hyper_search(build_fn, data_param, params)

                exp_dir = experiment_dir(data_param)
                result.save(exp_dir)

                print result
            except Exception as e:
                print e

    def hyper_search(self, build_fn, data_param, params):

        x_train, y_train, x_test, y_test = data_utils.get_train_test_data(**data_param)

        runner = Runner(build_fn, x_train, y_train, self.cv_splits, self.runs, params)

        # res, detail = self.run_solver(runner.run, runner.get_params())
        optimizer = PSOptimizer(4, 10)
        res = optimizer.optimize(runner.run, params)

        # print detail.stats
        print runner.get_log()
        print runner.get_best_params()
        print runner.get_best_results()

        # # Save results
        # try:
        #     if isinstance(best_params, list):
        #         result = MultiResult(model_wrapper, log, best_params, stats, data_param['country'])
        #     else:
        #         result = SingleResult(model_wrapper, log, best_params, stats, data_param['country'])
        # except Exception as e:
        #     print e
        #
        # return result
