import warnings
warnings.filterwarnings("ignore")

from mlp import mlp

from collections import OrderedDict
from src.utils.data_utils import VARIABLES, COUNTRIES
import argparse

from src.neuralnets.hypersearch import HyperSearch, var, choice



one_one = [([i], [i]) for i in VARIABLES]
all_one = [(VARIABLES, [i]) for i in VARIABLES]
all_all = [(VARIABLES, VARIABLES)]
data_vars = {'one_one': one_one,
             'many_one': all_one,
             'many_many': all_all}


def main(args):
    data_params = OrderedDict()
    data_params['country'] = [args['country']]
    data_params['vars'] = data_vars[args['vars']]
    data_params['lags'] = [args['lags']]

    params = OrderedDict()
    params['neurons'] = choice([var(1, 8, int)],
                               [var(1, 8, int), var(1, 8, int)])
    params['epochs'] = var(50, 300, int)
    params['batch_size'] = var(5, 20, int)
    # params['input_size'] = var(1, 16, int)

    searcher = HyperSearch(solver='pso', num_particles=7, num_generations=7, output_dir='mlp_experiments', cv_splits=3)

    searcher.hyper_data_search(mlp, data_params, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('country', choices=['EA', 'US'])
    parser.add_argument('vars', choices=['one_one', 'many_one', 'many_many'])
    parser.add_argument('lags', type=int)
    args = parser.parse_args()
    args = vars(args)
    # args = {'country': 'EA', 'vars': 'many_many', 'lags': 4}
    main(args)
