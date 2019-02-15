import pandas as pd
import numpy as np
import datetime as dt
import logging


from . import constants as constant
from . import utils as util
from . import retrieve_data as rd

import os
import itertools


COLUMN_SEQUENCE = ['M_t-1','H_t','W_t','M_t']


nWeeks = 2
nHours = 48
nStates = 2

def generate_map_df(path,SAVE):
    """
    Generate an encoded feature column
    """
    s = [np.arange(0,nHours),np.arange(0,nWeeks),np.arange(0,nStates)]
    data = list(itertools.product(*s))
    df_map = pd.DataFrame(columns=['H_t','W_t','M_t'],data=data)
    df_map['encoded'] = df_map.apply(lambda x:x['W_t'] + nWeeks*x['H_t']+ nWeeks*nHours*x['M_t'] +1 ,axis=1)
    if SAVE:
        df_map.to_csv(os.path.join(path,"hmm_encoding.csv"))
    return df_map

def test_data_validity_for_train_test(test_array, train_array):
    """
    Test for existance of data following cleaning
    """
    for array_ in [test_array, train_array]:
        if array_.shape[0]==0:
            return False
    if train_array['M_t'].value_counts().shape[0]!=2:
        return False
    return True

class hmm_test():
    """
    A class to generate input data for the HMM testing which needs to be run in Matlab
    """
    def __init__(self, tstat_id, save, save_path):
        """
        Initialize the class
        """
        self.thermostat = rd.thermostat_data(tstat_id)
        self.thermostat.main()
        self.data = self.thermostat.df_clean.copy()
        self.data = self.data.dropna(axis=0)
        self.data = self.data.astype('int')
        self.data['encoded'] = self.data.apply(lambda x:x['W_t'] + nWeeks*x['H_t']+ nWeeks*nHours*x['M_t'] +1 ,axis=1)
        self.SAVE = save
        self.SAVE_PATH = save_path
        
        self.encoded = generate_map_df(self.SAVE_PATH, self.SAVE)


    def run(self):
        """
        Run the file generation scripts
        """
        logging.info("thermostat {} - main execution for HMM run".format(self.thermostat.tstat_id))
        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))
        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))

            self.train_test_split(test_days[test_case])
            logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))

            if test_data_validity_for_train_test(self.test, self.train):
                self.save_train_and_test(test_case)

    def train_test_split(self, split_day):
        """
        Split the data for train and testing
        """
        self.train, self.test = util.make_train_and_test_dataframes(self.data, split_day, time_step_buffer=0)

    def save_train_and_test(self,test_case):
        """
        Save the train and test data for use by the Matlab scripts
        """
        self.train.to_csv(os.path.join(self.SAVE_PATH,'data/train/train_{}_{}.csv'.format(test_case, 
                            self.thermostat.tstat_id)))
        self.test.to_csv(os.path.join(self.SAVE_PATH,'data/test/test_{}_{}.csv'.format(test_case,
                             self.thermostat.tstat_id)))
