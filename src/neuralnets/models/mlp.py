from collections import OrderedDict
from src.utils.data_utils import VARIABLES, COUNTRIES

from keras.layers import Dense

from src.neuralnets.forecast_model.forecast_models import ForecastModel, create_input_layers, create_output_layers
from src.neuralnets.hypersearch import HyperSearch, var, choice


def mlp(num_inputs, num_outputs, input_size, neurons):
    input, layer = create_input_layers(num_inputs, input_size)

    for i in neurons:
        layer = Dense(i, activation='relu')(layer)
    output = create_output_layers(num_outputs, layer)

    model = ForecastModel(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model


def main():
    data_params = OrderedDict()
    data_params['country'] = COUNTRIES
    data_params['vars'] = [([i], [i]) for i in VARIABLES]
    data_params['vars'].extend([(VARIABLES, [i]) for i in VARIABLES])
    data_params['vars'].extend([(VARIABLES, VARIABLES)])
    data_params['lags'] = [4, 8]
    print data_params
    params = OrderedDict()
    params['neurons'] = choice([var(1, 8, int)],
                               [var(1, 8, int), var(1, 8, int)])
    params['epochs'] = var(50, 300, int)
    params['batch_size'] = var(5, 20, int)

    searcher = HyperSearch(solver='pso', num_particles=6, num_generations=6)

    searcher.hyper_data_search(mlp, data_params, params)


if __name__ == '__main__':
    main()
