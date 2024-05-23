# Allostatic Control of Persistent States in Spiking Neural Networks for perception and computation

## Code Pre-Requisites

The following code requires usage of Nest Simulator. During development, the code uses nest 3.7. The full documentation and download links can be found here:

General website: ```https://nest-simulator.org/```

Download link: ```https://nest-simulator.readthedocs.io/en/stable/installation/index.html```

Documentation: ```https://nest-simulator.readthedocs.io/en/stable/```

## Running Guidelines

### Data Collection of Model

The following part of the code runs the model and collect data from it for analysis work. Collected data of the sample run for gain modulation membrane time constant, {900, 1000} has been provided. These have been tested for number {1,2,3,4} that we have used in the model. This has been run over a timestep of 50,000 with random seeds from the nest simulator. New data can be collected by running the following script, subject to the pre-requistes being met:
```
python3 run_model.py
```
Data will be stored in ```/exp_data/<gain_mod_membrane_time_constant_value>_tau/data_spike_<number>.py```. If new data are being collected, these will overwrite the old files.

### Data Analysis of Model

