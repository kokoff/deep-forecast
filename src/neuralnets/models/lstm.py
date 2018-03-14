import warnings

warnings.filterwarnings("ignore")

from collections import OrderedDict

from keras.layers import Dense, Reshape, LSTM, Input
from keras.layers.wrappers import TimeDistributed
from src.neuralnets.forecast_model.forecast_models import ForecastModel, create_input_layers, create_output_layers, \
    RecurrentModel
from src.neuralnets.forecast_model.forecast_model_wrapper import ForecastRegressor


def lstm(num_inputs, num_outputs, input_size, neurons):
    # input, layer = create_input_layers(num_inputs, input_size)
    input = Input(batch_shape=(1, 1,))

    layer = Reshape((1, 1))(input)
    for i, n in enumerate(neurons):
        seq = i != (len(neurons)-1)
        layer = LSTM(n, stateful=True, return_sequences=seq)(layer)
    layer = Dense(1, activation='linear')(layer)

    model = RecurrentModel(inputs=input, outputs=layer)
    model.compile(optimizer='adam', loss='mse')

    return model


def main():
    from matplotlib import pyplot as plt
    params = OrderedDict()
    params['neurons'] = 10
    params['input_size'] = 1
    params['epochs'] = 1
    params['batch_size'] = 1

    data_params = OrderedDict()
    data_params['country'] = 'EA'
    data_params['vars'] = (['CPI'], ['CPI'])
    data_params['lags'] = 1

    wrapper = ForecastRegressor(lstm, data_params, params)
    wrapper.set_params(**params)
    print wrapper.fit()
    print wrapper.predict('val')
    print wrapper.forecast('val')
    print wrapper.evaluate_losses(2)
    predictions, forecasts = wrapper.get_predictions_and_forecasts()

    predictions[0].plot()
    plt.show()

    forecasts[0].plot()
    plt.show()


if __name__ == '__main__':
    main()
