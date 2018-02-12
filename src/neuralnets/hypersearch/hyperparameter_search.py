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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Validator:
    def __init__(self, f, x_train, y_train, x_val, y_val, runs, cross_validation=False):
        self.x = pd.concat([x_train, x_val])
        self.y = pd.concat([y_train, y_val])
        self.runs = runs
        self.build_fn = f
        if not cross_validation:
            l1 = [i for i in range(len(x_train))]
            l2 = [i for i in range(len(x_val))]
            self.split = [(l1, l2) for i in range(runs)]
        else:
            splitter = TimeSeriesSplit(runs)
            self.split = list(splitter.split(self.x, self.y))

        self.data = []
        for i, j in self.split:
            temp = []
            temp.append(self.x.iloc[i])
            temp.append(self.y.iloc[i])
            temp.append(self.x.iloc[j])
            temp.append(self.y.iloc[j])
            self.data.append(temp)

    def validate(self, **params):

        result = map(lambda data: self.build_fn(*data, **params), self.data)

        val_result = [i[0] for i in result]
        train_result = [i[1] for i in result]

        val_avg = np.average(val_result)
        train_avg = np.average(train_result)

        return val_avg, train_avg


class NormalValidator:
    def __init__(self, model_wrapper, runs, pmap=map):
        self.pmap = pmap
        self.runs = runs
        self.model_wrapper = model_wrapper

    def validation_run(self, **params):
        self.model_wrapper.set_params(**params)
        self.model_wrapper.fit(verbose=0)
        eval = self.model_wrapper.evaluate('val')
        train = self.model_wrapper.evaluate('train')
        return eval, train

    def validate(self, **params):
        self.model_wrapper.set_params(**params)

        result = self.pmap(lambda param: self.validation_run(**param), [params] * self.runs)

        val_result = [i[0] for i in result]
        train_result = [i[1] for i in result]

        if isinstance(val_result[0], list):
            val_avg = np.array(val_result).mean(axis=0).tolist()
            train_avg = np.array(train_result).mean(axis=0).tolist()
        else:
            val_avg = np.average(val_result)
            train_avg = np.average(train_result)

        return val_avg, train_avg


class Runner:

    def __init__(self, model_wrapper, runs, pmap=map):
        self.log = OrderedDict()
        self.log_initialised = False
        self.validator = NormalValidator(model_wrapper, runs, pmap)

    def init_log(self, params):
        for key in params:
            self.log[key] = []
        self.log.update({'train': [], 'val': []})
        self.log_initialised = True

    def run(self, **params):
        K.clear_session()
        if not self.log_initialised:
            self.init_log(params)

        val, train = self.validator.validate(**params)

        print params, val, train

        self.log['val'].append(val)
        self.log['train'].append(train)
        for param, value in params.iteritems():
            self.log[param].append(value)

        if not isinstance(val, list):
            return val
        else:
            return val

    def get_best_params(self):
        if isinstance(self.log['val'][0], list):
            best_paramss = []
            for j in range(len(self.log['val'][0])):
                val = [i[j] for i in self.log['val']]
                best_index = val.index(min(val))
                best_params = OrderedDict()
                for key, value in self.log.iteritems():
                    if key not in ['train', 'val']:
                        best_params[key] = value[best_index]
                best_paramss.append(best_params)
            return best_paramss
        else:
            val = self.log['val']
            best_index = val.index(min(val))
            best_params = OrderedDict()
            for key, value in self.log.iteritems():
                if key not in ['train', 'val']:
                    best_params[key] = value[best_index]
            return best_params


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
    # elif solver == 'nelder-mead':
    #     ftol = params.pop('ftol', 0.0001)
    #     max_iter = params.pop('max_iter', None)
    #     solver = NelderMead(ftol, max_iter, **params)
    # elif solver == 'cma-es':
    #     num_generations = params.pop('num_generations', 100)
    #     sigma = params.pop('sigma', 1.0)
    #     Lambda = params.pop('lambda', None)
    #     solver = CMA_ES(num_generations, sigma, Lambda, **params)
    else:
        raise ValueError(solver, 'is not a valid solver.')
    return solver


class HyperSearch:
    def __init__(self, solver, parallel, validation_runs, **kwargs):
        self.solver_name = solver
        self.solver_params = kwargs
        self.parallel = parallel
        self.runs = validation_runs

    def hyper_data_search(self, build_fn, data_params_dict, params):
        data_params = ParameterGrid(data_params_dict)
        for data_param in data_params:
            print self.hyper_search(build_fn, data_param, params)

    def hyper_search(self, build_fn, data_param, params):
        '''Perform hyperparameter search using optunity as a backend. Possible solvers include
        grid search     args: None
        random search   args: num_evals
        particle swarm  args: num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0
        tpe             args: num_evals=100, seed=None
        sobol           args: num_evals, seed=None, skip=None
        nelder-mead     args: ftol=0.0001, max_iter=None,

        cma-es          args: num_generations, sigma=1.0, Lambda=None

        '''

        model_wrapper = ModelWrapper(build_fn)
        model_wrapper.set_data_params(**data_param)

        pmap = optunity.pmap if self.parallel else map

        validator = Runner(model_wrapper, self.runs, pmap=pmap)

        solver = get_solver(self.solver_name, self.solver_params, params)
        res, detail = optunity.optimize(solver, validator.run, maximize=False, pmap=map)

        print res
        print detail
        log = validator.log
        best_params = validator.get_best_params()
        stats = detail.stats

        # Save results
        if isinstance(best_params, list):
            result = MultiResult(model_wrapper, log, best_params, stats)
        else:
            result = SingleResult(model_wrapper, log, best_params, stats)

        result.save('temp')

        return result
