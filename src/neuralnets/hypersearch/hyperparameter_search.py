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
from sklearn.model_selection import ParameterGrid
import datetime

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


class ModelEvaluator:
    def __init__(self, wrapper, best_params):
        self.wrapper = wrapper
        self.best_params = best_params

    def eval_performance(self, num_runs):
        performances = []
        if isinstance(self.best_params, list):
            for i, params in enumerate(self.best_params):
                self.wrapper.set_params(**params)
                results = OrderedDict()
                results['pred train'] = []
                results['pred val'] = []
                results['pred test'] = []
                results['fcast train'] = []
                results['fcast val'] = []
                results['fcast test'] = []

                for j in range(num_runs):
                    self.wrapper.reset()
                    results['pred train'].append(self.wrapper.evaluate('train', verbose=0)[i])
                    results['pred val'].append(self.wrapper.evaluate('val', verbose=0)[i])
                    results['pred test'].append(self.wrapper.evaluate('test', verbose=0)[i])
                    results['fcast train'].append(self.wrapper.evaluate_forecast('train', verbose=0)[i])
                    results['fcast val'].append(self.wrapper.evaluate_forecast('val', verbose=0)[i])
                    results['fcast test'].append(self.wrapper.evaluate_forecast('test', verbose=0)[i])

                for key, value in results.iteritems():
                    results[key] = np.mean(value, axis=0).tolist()
                performances.append(pd.DataFrame(results, index=[0]))
            return performances
        else:
            self.wrapper.set_params(**self.best_params)
            results = OrderedDict()
            results['pred train'] = []
            results['pred val'] = []
            results['pred test'] = []
            results['fcast train'] = []
            results['fcast val'] = []
            results['fcast test'] = []
            for i in range(num_runs):
                self.wrapper.reset()
                results['pred train'].append(self.wrapper.evaluate('train', verbose=0))
                results['pred val'].append(self.wrapper.evaluate('val', verbose=0))
                results['pred test'].append(self.wrapper.evaluate('test', verbose=0))
                results['fcast train'].append(self.wrapper.evaluate_forecast('train', verbose=0))
                results['fcast val'].append(self.wrapper.evaluate_forecast('val', verbose=0))
                results['fcast test'].append(self.wrapper.evaluate_forecast('test', verbose=0))

            for key, value in results.iteritems():
                results[key] = np.mean(value, axis=0).tolist()
            return pd.DataFrame(results, index=[0])

    def eval_prediction(self):
        if not isinstance(self.best_params, list):
            self.wrapper.set_params(**self.best_params)
        else:
            self.wrapper.set_params(**self.best_params[0])

        predictions = []
        results = OrderedDict()

        train = self.wrapper.predict('train')
        val = self.wrapper.predict('val')
        test = self.wrapper.predict('test')
        results['prediction'] = pd.concat([train, val, test])

        train = self.wrapper.forecast('train')
        val = self.wrapper.forecast('val')
        test = self.wrapper.forecast('test')
        results['forecast'] = pd.concat([train, val, test])

        predictions.append(results)
        if not isinstance(self.best_params, list):
            return predictions[0]
        else:
            for i, params in enumerate(self.best_params[1:]):
                results = OrderedDict()
                self.wrapper.set_params(**params)

                train = self.wrapper.predict('train').iloc[:, i]
                val = self.wrapper.predict('val').iloc[:, i]
                test = self.wrapper.predict('test').iloc[:, i]
                results['prediction'] = pd.concat([train, val, test])

                train = self.wrapper.forecast('train').iloc[:, i]
                val = self.wrapper.forecast('val').iloc[:, i]
                test = self.wrapper.forecast('test').iloc[:, i]
                results['forecast'] = pd.concat([train, val, test])

                predictions.append(results)
        return predictions


class MultiResult:
    def __init__(self, wrapper, log, params, stats):
        self.params = [pd.DataFrame(param, index=[0]) for param in params]
        self.log = self.logs2dfs(log)
        self.stats = pd.DataFrame(stats, index=[str(datetime.datetime.now())])
        evaluator = ModelEvaluator(wrapper, params)
        self.performance = evaluator.eval_performance(2)
        self.predictions = evaluator.eval_prediction()
        self.vars = ['total'] + self.predictions[0].values()[0].columns.levels[0].tolist()
        assert len(self.vars) == len(self.params)
        self.results = []

        for i, variable in enumerate(self.vars):
            self.results.append(
                Result(self.log[i], self.params[i], self.stats, self.performance[i], self.predictions[i], variable))

    def __str__(self):
        string = ''
        for i in self.results:
            string += str(i) + '\n'
        return string

    def save(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

        for result in self.results:
            result.save(dir)

    def logs2dfs(self, log):
        logs = []
        for i in range(len(self.params)):
            new_log = OrderedDict()
            for key, value in log.iteritems():
                if not isinstance(value, list):
                    new_log[key] = value
                else:
                    new_log[key] = value[i]
            logs.append(pd.DataFrame(new_log))
        return logs


class SingleResult:
    def __init__(self, wrapper, log, params, stats):
        self.log = pd.DataFrame(log)
        self.stats = pd.DataFrame(stats, index=[str(datetime.datetime.now())])
        self.params = pd.DataFrame(params, index=[0])
        evaluator = ModelEvaluator(wrapper, params)
        self.performance = evaluator.eval_performance(20)
        self.predictions = evaluator.eval_prediction()
        self.var = self.predictions.values()[0].columns.levels[0].tolist()[0]
        self.result = Result(self.log, self.params, self.stats, self.performance, self.predictions, self.var)

    def __str__(self):
        return str(self.result)

    def save(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.result.save(dir)


class Result:
    def __init__(self, log, params, stats, performance, predictions, var):
        self.log = log
        self.stats = stats
        self.params = params
        self.performance = performance
        self.predictions = predictions
        self.var = var

    def __str__(self):
        string = ''
        string += '------------------------------------------------------------------------------\n'
        string += 'Result: ' + self.var + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.log) + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.stats) + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.params) + '\n'
        string += '------------------------------------------------------------------------------\n'
        string += str(self.performance) + '\n'
        string += '------------------------------------------------------------------------------\n\n'

        return string

    def save(self, dir):

        if not os.path.exists(dir):
            os.mkdir(dir)

        basedir = os.path.join(dir, self.var)
        if not os.path.exists(basedir):
            os.mkdir(basedir)

        self.log.to_csv(os.path.join(basedir, 'log.csv'))
        self.stats.to_csv(os.path.join(basedir, 'stats.csv'), mode='a')
        self.params.to_csv(os.path.join(basedir, 'parameters.csv'))
        self.performance.to_csv(os.path.join(basedir, 'performance.csv'))
        for key, value in self.predictions.iteritems():
            value.to_csv(os.path.join(basedir, key + '.csv'))
