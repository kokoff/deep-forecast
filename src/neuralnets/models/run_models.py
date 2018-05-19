import warnings

warnings.filterwarnings("ignore")

from mlp import mlp
from lstm import lstm

from collections import OrderedDict
from src.utils.data_utils import VARIABLES, COUNTRIES
import argparse

from src.neuralnets.hypersearch import HyperSearch, var, choice
from src.neuralnets.hypersearch.results.visualisation import produce_plots


def lstm_experiment(args):
    data_params = OrderedDict()
    data_params['country'] = [args['country']]
    data_params['vars'] = [(args['in'], args['out'])]

    params = OrderedDict()
    params['neurons'] = choice([var(1, 15, int)],
                               [var(1, 15, int), var(1, 15, int)],
                               [var(1, 15, int), var(1, 15, int), var(1, 15, int)])
    params['dropout'] = var(0.0, 0.5, float)
    params['epochs'] = var(10, 50, int)
    params['batch_size'] = 1
    params['input_size'] = 1

    output_dir = 'lstm_experiments'
    searcher = HyperSearch(solver='pso', num_particles=5, num_generations=5, output_dir=output_dir, cv_splits=5,
                           eval_runs=10)

    searcher.hyper_data_search(lstm, data_params, params)
    produce_plots(output_dir)


def mlp_experiment(args):
    data_params = OrderedDict()
    data_params['country'] = [args['country']]
    data_params['vars'] = [(args['in'], args['out'])]

    params = OrderedDict()
    params['neurons'] = choice([var(1, 15, int)],
                               [var(1, 15, int), var(1, 15, int)])
    params['dropout'] = var(0.0, 0.5, float)
    params['epochs'] = var(10, 150, int)
    params['batch_size'] = var(1, 10, int)
    params['input_size'] = var(4, 10, int)

    output_dir = 'mlp_experiments'
    searcher = HyperSearch(solver='pso', num_particles=5, num_generations=5, output_dir=output_dir, cv_splits=5,
                           eval_runs=10)

    searcher.hyper_data_search(mlp, data_params, params)
    produce_plots(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['mlp', 'lstm'])
    parser.add_argument('-c', '--country', choices=['EA', 'US'])
    parser.add_argument('-i', '--in', nargs='*')
    parser.add_argument('-o', '--out', nargs='*')
    args = parser.parse_args()
    args = vars(args)
    print args
    # args = {'model': 'mlp', 'country': 'EA', 'vars': 'one_one', 'lags': 1, 'in': ['CPI'], 'out': ['CPI']}

    if args['model'] == 'lstm':
        print 'LSTM experiment'
        lstm_experiment(args)
    elif args['model'] == 'mlp':
        print 'MLP experiment'
        mlp_experiment(args)
    else:
        print 'Invalid Model'
