from collections import OrderedDict

from keras.layers import Dense

from src.neuralnets.forecast_model.forecast_models import ForecastModel, create_input_layers, create_output_layers
from src.neuralnets.hypersearch import HyperSearch


def makeModel(num_inputs, input_size, num_outputs, output_size, neurons, activation='relu'):
    input, layer = create_input_layers(num_inputs, input_size)

    if isinstance(list, neurons):
        for i in neurons:
            output = create_output_layers(num_outputs, output_size, layer)
            layer = Dense(neurons, activation=activation)(layer)

    model = ForecastModel(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss='mse')

    return model
