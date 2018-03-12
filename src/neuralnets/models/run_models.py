import warnings

warnings.filterwarnings("ignore")

from mlp import mlp

from collections import OrderedDict
from src.utils.data_utils import VARIABLES, COUNTRIES
import argparse

from src.neuralnets.hypersearch import HyperSearch, var, choice
from src.neuralnets.hypersearch.results.visualisation import produce_plots

one_one = [([i], [i]) for i in VARIABLES]
all_one = [(VARIABLES, [i]) for i in VARIABLES]
all_all = [(VARIABLES, VARIABLES)]
data_vars = {'one_one': one_one,
             'many_one': all_one,
             'many_many': all_all}


def main(args):
    data_params = OrderedDict()
    data_params['country'] = ['EA']
    data_params['vars'] = [(['CPI'], ['CPI'])]
    data_params['lags'] = [args['lags']]

    params = OrderedDict()
    params['neurons'] = choice([var(1, 15, int)],
                               [var(1, 15, int), var(1, 15, int)])
    params['epochs'] = var(50, 1500, int)
    params['batch_size'] = 2
    # params['input_size'] = var(1, 16, int)

    searcher = HyperSearch(solver='pso', num_particles=5, num_generations=5, output_dir='mlp_experiments', cv_splits=4, eval_runs=2)

    searcher.hyper_data_search(mlp, data_params, params)
    produce_plots('mlp_experiments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('country', choices=['EA', 'US'])
    parser.add_argument('vars', choices=['one_one', 'many_one', 'many_many'])
    parser.add_argument('lags', type=int)
    args = parser.parse_args()
    args = vars(args)
    args = {'country': 'EA', 'vars': 'one_one', 'lags': 4}
    main(args)
