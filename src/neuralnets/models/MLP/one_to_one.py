import sys
print sys.path
from collections import OrderedDict

from keras.layers import Dense

from src.neuralnets.forecast_model.forecast_models import ForecastModel, create_input_layers, create_output_layers
from src.neuralnets.hypersearch import HyperSearch
from src.utils import data_utils




def makeModel(neurons):
    input, layer = create_input_layers(1, 4)
    layer = Dense(neurons, activation='relu')(layer)
    output = create_output_layers(1, 1, layer)

    model = ForecastModel(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss='mse')

    return model


DATA_PARAMS = OrderedDict()
DATA_PARAMS['country'] = ['EA', 'US']
DATA_PARAMS['vars'] = [([i], [i]) for i in data_utils.VARIABLES]
DATA_PARAMS['lags'] = [(4, 1)]


def main():
    params = OrderedDict()
    params['neurons'] = (int, 1, 10)
    params['epochs'] = (int, 50, 250)
    params['batch_size'] = (int, 1, 10)

    searcher = HyperSearch(solver='rso', num_runs=10)

    searcher.hyper_data_search(makeModel, DATA_PARAMS, params)


if __name__ == '__main__':
    main()
