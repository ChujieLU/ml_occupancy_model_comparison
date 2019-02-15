# ml_occupancy_model_comparison

The code in this repositiory was the basis to the paper 'Comparison of machine learning models for occupancy prediction in residential buildings using connected thermostat data'. The analysis was conducted using the ecobee Donate Your Data (DYD) dataset. In particular it was the 2017 release of data. Subsequent releases have changed some of the implmentation notes. 

To be able to run all the models you will need (along with other standard python packages):
* sckikit-learn
* pgmpy
* tensorflow
* Kevin Murphy's Matlab methods for hidden Markov models

Constants (such as the data directory) are set in 'constants.py'

Each method can be run independently using the 'run_test.py' function. In that function many hyperparameter values (depending on the model) can also be set prior to running. 
