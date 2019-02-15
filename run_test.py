from test_framework import lr_encoded_test as lre
from test_framework import rf_framework as rf
from test_framework import mm_framework_v2 as mm
from test_framework import rnn_framework_v4 as rnn
from test_framework import hmm_framework as hmm
from test_framework import baseline_framework as base
from test_framework import constants as const
from test_framework import utils as util

import click
import os

@click.command()
@click.option('--model', type=click.Choice(['LR', 'RF', 'MM','RNN', 'HMM','Base']), help="Which test you want to run")
@click.option('--test_name', help="What do you want to save this run as?")
@click.option('--data_dir', help="Location of data direcory", default=const.RAW_DATA_DIRECTORY)
@click.option('--save', is_flag=True, help='Binary flag to save results')
@click.option('--time', is_flag=True, default=False, help='Binary flag to time traning and inference')
def start(model, test_name, data_dir, save, time):
    
    click.echo('The model run is %s!' % model)
    click.echo('The test is to be called %s!' % test_name)
    click.echo('Data will be taken from %s!' % data_dir)

    save_path = os.path.join('.',test_name,model)
    validate_output_directory(save_path)

    tstat_list = thermostat_list()

    if model == 'LR':
        run_LR_thermostat(tstat_list, model, 6, save, save_path, time)

    elif model == 'RF':
        run_RF_thermostat(tstat_list, model, 6, save, save_path)
        
    elif model == 'MM':
        run_MM_thermostat(tstat_list, model, 6, 1, save, save_path)

    elif model == 'RNN':
        run_RNN_thermostat(tstat_list, model, save, save_path)

    elif model == 'HMM':
        click.echo('This is will only format the data for later use by Matlab.')
        run_HMM_thermostat(tstat_list, model, save, save_path)
    
    elif model == 'Base':
        run_BASE_thermostat(tstat_list, model, 6, save, save_path)

def validate_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def thermostat_list():
    tstat_list = []
    for tstat in util.files(const.RAW_DATA_DIRECTORY):
        tstat_list.append(tstat)
    return tstat_list
    

def run_LR_thermostat(files, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH, TIME):
    for file in files:
        tstat = util.retrieve_tstat_id(file)
        train = lre.lr_extended(tstat, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH, TIME)
        train.run()

def run_RF_thermostat(files, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH):
    for file in files:
        tstat = util.retrieve_tstat_id(file)
        train = rf.rf_test(tstat, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH)
        train.run()

def run_MM_thermostat(files, METHOD, N_STEPS_AHEAD, N_STEPS_BEHIND, SAVE, SAVE_PATH):
    for file in files:
        tstat = util.retrieve_tstat_id(file)
        train = mm.mm_test(tstat, METHOD, N_STEPS_AHEAD, N_STEPS_BEHIND, SAVE, SAVE_PATH)
        train.run()

def run_RNN_thermostat(files, METHOD, SAVE, SAVE_PATH):
    N_BATCH = 50
    N_EPOCH = 5
    INPUT_LAYERS = 2
    HIDDEN_LAYER_SIZE = 8
    DROPOUT = 0.2
    for file in files:
        tstat = util.retrieve_tstat_id(file)
        train = rnn.rnn_test( tstat_id = tstat, save_path=SAVE_PATH, method_name=METHOD,
                              n_batch=N_BATCH, n_epoch=N_EPOCH, 
                              input_layers=INPUT_LAYERS, hidden_layer_size=HIDDEN_LAYER_SIZE,
                              dropout=DROPOUT, save=SAVE, plot=False)
        train.run()

def run_HMM_thermostat(files, METHOD, SAVE, SAVE_PATH):  
    validate_output_directory(os.path.join(SAVE_PATH,"data","train"))
    validate_output_directory(os.path.join(SAVE_PATH,"data","test"))
    for file in files:
        tstat = util.retrieve_tstat_id(file)
        train = hmm.hmm_test(tstat, SAVE, SAVE_PATH)
        train.run()

def run_BASE_thermostat(files, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH):
    for file in files:
        tstat = util.retrieve_tstat_id(file)
        train = base.base_tests(tstat, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH)
        train.run()

if __name__ == "__main__":
    start()