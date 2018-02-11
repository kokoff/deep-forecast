import functools
from collections import OrderedDict

import numpy as np
import optunity
import pandas as pd

from src.utils import data_utils
from keras import backend as K
from optunity.solvers import GridSearch, RandomSearch, ParticleSwarm, TPE, Sobol, NelderMead, CMA_ES
from sklearn.model_selection import TimeSeriesSplit
from multiprocessing import Pool
from forecast_models import ModelWrapper
from scoop import futures
import os
import tensorflow as tf
import csv

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
    def __init__(self, f, x_train, y_train, x_val, y_val, runs, cross_validation=False, pmap=map):
        self.pmap = pmap
        self.runs = runs
        # self.build_fn = functools.partial(f, x_train, y_train, x_val, y_val)
        self.model = ModelWrapper(f)
        x_train, y_train, x_val, y_val, x_test, y_test = data_utils.get_data_formatted('EA',
                                                                                       {'x': ['CPI', 'GDP', 'UR'],
                                                                                        'y': ['CPI', 'GDP']},
                                                                                       4, 1,
                                                                                       12, 12)
        self.model.set_data(x_train, y_train, x_val, y_val, x_test, y_test)

    def run(self, **params):
        self.model.set_params(**params)
        self.model.fit(verbose=0)
        eval = self.model.evaluate('val')
        train = self.model.evaluate('train')
        return eval, train

    def validate(self, **params):
        self.model.set_params(**params)

        result = self.pmap(lambda param: self.run(**param), [params] * self.runs)

        val_result = [i[0] for i in result]
        train_result = [i[1] for i in result]

        val_avg = np.average(val_result)
        train_avg = np.average(train_result)

        return val_avg, train_avg


class Runner:

    def __init__(self, f, x_train, y_train, x_val, y_val, runs, cv=False, pmap=map):
        self.log = OrderedDict()
        self.log_initialised = False
        self.build_f = f
        self.validator = NormalValidator(f, x_train, y_train, x_val, y_val, runs, cv, pmap)

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

        return val

    def log2dataframe(self):
        return pd.DataFrame(self.log)

    def get_best_params(self):
        index = self.log['val'].index(max(self.log['val']))
        best_params = OrderedDict()
        for key, value in self.log.iteritems():
            if key not in ['train', 'val']:
                best_params[key] = value[index]
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

    def hyper_search(self, make_fn, eval_fn, **params):
        '''Perform hyperparameter search using optunity as a backend. Possible solvers include
        grid search     args: None
        random search   args: num_evals
        particle swarm  args: num_particles, num_generations, max_speed=None, phi1=1.5, phi2=2.0
        tpe             args: num_evals=100, seed=None
        sobol           args: num_evals, seed=None, skip=None
        nelder-mead     args: ftol=0.0001, max_iter=None,

        cma-es          args: num_generations, sigma=1.0, Lambda=None

        '''
        x_train, y_train, x_val, y_val, x_test, y_test = data_utils.get_data_formatted('EA',
                                                                                       {'x': ['CPI'], 'y': ['CPI']},
                                                                                       4, 1,
                                                                                       12, 12)
        pmap = optunity.pmap if self.parallel else map

        validator = Runner(eval_fn, x_train, y_train, x_val, y_val, self.runs, pmap=pmap)

        solver = get_solver(self.solver_name, self.solver_params, params)
        res, detail = optunity.optimize(solver, validator.run, maximize=False, pmap=map)

        best_params = validator.get_best_params()
        make_fn = functools.partial(make_fn, x_train, y_train, x_val, y_val, **best_params)
        best_performance = eval_performance(make_fn, x_train, y_train, x_val, y_val, x_test, y_test)
        best_prediction = eval_prediction(make_fn, x_train, x_val, x_test)

        result = Result(make_fn=make_fn, log=validator.log, time=detail.stats['time'],
                        params=validator.get_best_params(),
                        performance=best_performance, predictions=best_prediction)

        result.save('temp')

        return result


def eval_performance(make_fn, x_train, y_train, x_val, y_val, x_test, y_test, num_runs=20):
    results = OrderedDict()
    results['train'] = []
    results['val'] = []
    results['test'] = []
    for i in range(num_runs):
        model = make_fn()
        results['train'].append(model.evaluate(x_train, y_train, verbose=0))
        results['val'].append(model.evaluate(x_val, y_val, verbose=0))
        results['test'].append(model.evaluate(x_test, y_test, verbose=0))
    results['train'] = np.mean(results['train'])
    results['val'] = np.mean(results['val'])
    results['test'] = np.mean(results['test'])
    return results


def eval_prediction(make_fn, x_train, x_val, x_test):
    results = OrderedDict()
    model = make_fn()
    results['train'] = model.predict(x_train).flatten().tolist()
    results['val'] = model.predict(x_val).flatten().tolist()
    results['test'] = model.predict(x_test).flatten().tolist()
    return results


class Result:
    def __init__(self, make_fn, log, time, params, performance, predictions):
        self.make_fn = make_fn
        self.log = pd.DataFrame(log)
        self.time = time
        self.params = pd.DataFrame(params, index=[0])
        self.performance = pd.DataFrame(performance, index=[0])
        self.predictions = predictions

    def __str__(self):
        string = ''
        string += '\nRun Time: ' + self.time.__str__() + '\n'
        string += self.log.__str__() + '\n'
        string += '\nParameters:\n' + self.params.__str__() + '\n'
        string += '\nPerformance\n' + self.performance.__str__() + '\n'
        string += '\nPredictions\n'
        for i, j in self.predictions.iteritems():
            string += str(i) + ': ' + str(j) + '\n'
        self.model().summary()
        return string

    def model(self):
        return self.make_fn()

    def save(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

        self.log.to_csv(os.path.join(dir, 'log.csv'))
        with open(os.path.join(dir, 'run_stats.txt'), 'w') as f:
            f.write(str(self.time))
        self.params.to_csv(os.path.join(dir, 'parameters.csv'))
        self.performance.to_csv(os.path.join(dir, 'performance.csv'))


