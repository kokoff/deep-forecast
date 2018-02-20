from collections import OrderedDict

from keras.layers import Dense

from src.neuralnets.forecast_model.forecast_models import ForecastModel, create_input_layers, create_output_layers
from src.neuralnets.hypersearch import HyperSearch


def makeModel(num_inputs, input_size, num_outputs, output_size, neurons, optimiser='adam'):
    input, layer = create_input_layers(num_inputs, input_size)

    layer = Dense(neurons, activation='relu')(layer)

    output = create_output_layers(num_outputs, output_size, layer)

    model = ForecastModel(inputs=input, outputs=output)

    # optimizer = optimizers.get(optimiser)
    model.compile(optimizer='adam', loss='mse')

    return model


def main():
    data_params = OrderedDict()
    data_params['country'] = ['EA', 'US']
    data_params['vars'] = [
        (['CPI', 'GDP', 'UR', 'IR', 'LR10', 'LR10-IR', 'EXRATE'],
         ['CPI', 'GDP', 'UR', 'IR', 'LR10', 'LR10-IR', 'EXRATE'])
    ]
    data_params['lags'] = [(4, 1)]

    data_param = OrderedDict()
    data_param['country'] = 'EA'
    data_param['vars'] = (['CPI'],
                          ['CPI'])

    data_param['lags'] = (4, 1)

    params = OrderedDict()
    params['neurons'] = (int, 1, 50)
    params['epochs'] = (int, 100, 600)
    params['batch_size'] = (int, 1, 40)

    searcher = HyperSearch(solver='pso', cv_splits=5, validation_runs=2, eval_runs=10, num_particles=5,
                           num_generations=5)

    searcher.hyper_search(makeModel, data_param, params)


if __name__ == '__main__':
    main()
