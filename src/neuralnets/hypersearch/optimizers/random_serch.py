import timeit
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from utils import generate_sobol_sequences

import numpy as np

from search_space import param_decorator, SearchSpace


class RandomSearch:
    def __init__(self, num_runs, lb, ub, sobol):
        self.num_runs = num_runs
        self.lb = lb
        self.ub = ub
        self.sobol = sobol

        self.best_params = None
        self.best_result = np.inf

    def optimize(self, eval_fn):
        if self.sobol:
            params = generate_sobol_sequences(self.num_runs, self.lb, self.ub)
        else:
            params = [np.random.uniform(self.lb, self.ub) for i in range(self.num_runs)]

        for param in params:

            param = np.minimum(param, self.ub)
            param = np.maximum(param, self.lb)

            res = eval_fn(*param)
            if res < self.best_result:
                self.best_result = res
                self.best_params = param

        return self.best_params, self.best_result


class RSOptimizer:
    def __init__(self, num_runs, sobol=False):
        self.num_runs = num_runs
        self.sobol = sobol

    def optimize(self, run_f, params):
        search_tree = SearchSpace(params)

        lb = search_tree.get_lb()
        ub = search_tree.get_ub()
        f = param_decorator(run_f, search_tree)

        gs = RandomSearch(self.num_runs, lb, ub, self.sobol)

        start = timeit.default_timer()
        best_params, score = gs.optimize(f)
        end = timeit.default_timer() - start

        best_params = search_tree.transform(best_params)
        Result = namedtuple('Result', ['params', 'score', 'time'])

        return Result(best_params, score, end)


def func(x, y):
    print x, y
    return x ** 2 + y ** 2


def main():
    params = OrderedDict()
    params['x'] = (float, -10, 10)
    params['y'] = (float, -10, 10)

    opt = RSOptimizer(100, sobol=False)
    res = opt.optimize(func, params)
    print res.params
    print res.score
    print res.time


if __name__ == '__main__':
    main()
