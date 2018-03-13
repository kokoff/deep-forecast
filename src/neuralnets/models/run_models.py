import warnings

warnings.filterwarnings("ignore")

from mlp import mlp
from lstm import lstm

from collections import OrderedDict
from src.utils.data_utils import VARIABLES, COUNTRIES
import argparse

from src.neuralnets.hypersearch import HyperSearch, var, choice
from src.neuralnets.hypersearch.results.visualisation import produce_plots


def main(args):
    data_params = OrderedDict()
    data_params['country'] = [args['country']]
    data_params['vars'] = [(args['in'], args['out'])]
    # data_params['lags'] = [1]

    params = OrderedDict()
    params['neurons'] = choice([var(1, 15, int)],
                               [var(1, 15, int), var(1, 15, int)])
    params['epochs'] = var(10, 50, int)
    params['batch_size'] = var(1, 10, int)
    params['input_size'] = var(4, 10, int)

    searcher = HyperSearch(solver='pso', num_particles=5, num_generations=5, output_dir='mlp_experiments', cv_splits=4,
                           eval_runs=2)

    searcher.hyper_data_search(mlp, data_params, params)
    produce_plots('mlp_experiments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--country', choices=['EA', 'US'])
    parser.add_argument('-i', '--in', nargs='*')
    parser.add_argument('-o', '--out', nargs='*')
    args = parser.parse_args()
    args = vars(args)
    print args
    # args = {'country': 'EA', 'vars': 'one_one', 'lags': 1}
    main(args)
