# Forecasting Macroeconomic Parameters with Deep Learning Neural Networks

The project will make use of convolutional deep learning neural networks (as implemented through the Keras software library) to model macroeconomic data and forecast future developments.

The essence of the project is to apply a machine learning based approach to acquiring empirical forecasting models of macroeconomic variables, such as inflation, unemployment and GDP for one or more economies (e.g. UK, EU, US). The project will use Keras, a high-level neural networks API, written in Python and capable of running on top of either TensorFlow, CNTK or Theano. It was developed with a focus on enabling fast experimentation. The library supports both convolutional networks and recurrent networks, as well as combinations of the two, and it runs seamlessly on CPU and GPU.

An MEng student will be expected to produce models from data that represents more than one economy, using at least 2 macro-parameters in each case. The benefits of including key commodities, such as oil and gold, should also be at least discussed in the design section (if adequate data cannot be gathered). A certain level of automation of the experimental setup is expected to be developed, and discussed in the report.

----------

Reading:

Dimitar Kazakov and Tsvetomira Tsenova. January 2009. [Equation Discovery for Macroeconomic Modelling.](http://www-users.cs.york.ac.uk/~kazakov/papers/crc-Kazakov-Tsenova.pdf) International Conference on Agents and Artificial Intelligence, Porto, Portugal. 

Ranjit Bassi. [Forecasting Macroeconomic Data with Neural Networks.](https://www.cs.york.ac.uk/library/proj_files/2017/3rdyr/rb1217/rb1217.zip) 3rd year project, CS Dept, University of York. 2017

[Keras](https://keras.io): The Python Deep Learning library.


-----


## Usage

Set `EXPERIMENTS_DIR` in config.json. This is where experimental results will be stored.

Run ARIMA experiments:
```
python src/arima/arima.py
```


Run persistance baseline model:
```
python src/persistence/persistence.py
```

### Neural network experiments
In order to run the automatic model selection use:
```
python src/neuralnets/models/main.py 

usage: main.py [-h] -m {mlp,lstm,all} -d {yes,no,both} [-a] [-b] [-c]

optional arguments:
  -h, --help            show this help message and exit
  -m {mlp,lstm,all}, --model {mlp,lstm,all}
                        Which models to run.
  -d {yes,no,both}, --diff {yes,no,both}
                        Weather or not to use first order differencing
  -a, --one-to-one      This argument only applies to mlp. Run one-to-one
                        architecture
  -b, --many-to-one     This argument only applies to mlp. Run many-to-one
                        architecture
  -c, --many-to-many    This argument only applies to mlp. Run many-to-many
                        architecture
```

The search space of NN hyper parameters and hyper parameter optimisation algorithm can be changed in `src/neuralnets/models/run_models.py`. Default hyper parameter optimisation is Particle Swarm Optimisation. Alternatives are Grid Search Optimisation and Random Search Optimisation. 

Searched parameters can be added/removed in the model specific files 'src/neuralnets/models/mlp.py' and `src/neuralnets/models/lstm.py`. 

The hyperparameter optimisation package is more or less self contained and can be used independantly from the NN models. It can be found in `src/neuralnets/hypersearch/optimizers`. 

`src/neuralnets/forecast_model/` contains a scikit-learn like wrapper implementation of MLP and RNN models for forecasting time series. 



