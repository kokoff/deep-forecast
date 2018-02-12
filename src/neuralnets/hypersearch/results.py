import datetime
import os
from collections import OrderedDict

import numpy as np
import pandas as pd


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