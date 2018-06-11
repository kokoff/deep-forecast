# Forecasting Macroeconomic Parameters with Deep Learning Neural Networks

## Project Blog
The progress of the project was tracked via a Blog. The thesis and other delivarables can be found there: https://deepforecastproject.blogspot.com/


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



