import timeit
from collections import OrderedDict
from collections import namedtuple
from itertools import product

import numpy as np

from search_space import param_decorator, SearchSpace


class GridSearch:
    def __init__(self, lb, ub):
        params = []
        for l, u in zip(lb, ub):
            params.append(np.arange(l, u))
        self.args = product(*params)

        self.best_params = None
        self.best_result = np.inf

    def optimize(self, eval_fn):
        for param in self.args:
            res = eval_fn(*param)
            if res < self.best_result:
                self.best_result = res
                self.best_params = param
        return self.best_params, self.best_result


class GSOptimizer:
    def __init__(self):
        pass

    def optimize(self, run_f, params):
        search_tree = SearchSpace(params)

        lb = search_tree.get_lb()
        ub = search_tree.get_ub()
        f = param_decorator(run_f, search_tree)

        gs = GridSearch(lb, ub)

        start = timeit.default_timer()
        best_params, score = gs.optimize(f)
        end = timeit.default_timer() - start

        best_params = search_tree.transform(best_params)
        Result = namedtuple('Result', ['params', 'score', 'time'])

        return Result(best_params, score, end)


def func(x, y, pizza):
    print x, y, pizza
    return x ** 2 + y ** 2


def main():
    params = OrderedDict()
    params['x'] = [1, 2, 3, 4, 5, 6]
    params['y'] = {1: {'pizza': [1, 2, 3]}}

    opt = GSOptimizer()
    res = opt.optimize(func, params)
    print res.params
    print res.score
    print res.time


if __name__ == '__main__':
    main()
