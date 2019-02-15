

from . import lr_framework as lr_f
import os
import numpy as np
import pandas as pd
import itertools
import logging

import time

from . import constants as constant
from . import utils as util


def generate_map_df():
    """
    Generate a feature column and make this into the dummy variables of one-hot encoding
    """
    s=[ list(str(x) for x in np.arange(0,48)), [ '1', '0'], ['1','0']]
    data = list(itertools.product(*s))
    df_map = pd.DataFrame(columns=['H','W','M'],data=data)
    df_map['feature'] = df_map.apply(lambda x: str(x.H)+"_"+str(x.W)+"_"+str(x.M),axis=1)
    df_dummy = pd.get_dummies(df_map['feature'])
    df_dummy['feature'] = df_map['feature']
    return df_dummy, list(pd.get_dummies(df_map['feature']).columns)

def form_dataframe_to_arrays_encoded(df):
    X = df.filter(regex="\d_\d_\d").values
    Y = df.filter(regex="[\+]").values.flatten()
    return X,Y

class lr_extended(lr_f.lr_test):
    def __init__(self, tstat_id, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH, TIME):
        """
        Initialize the class and build from the general LR class
        """
        lr_f.lr_test.__init__(self, tstat_id)
        self.data = self.data.astype('int')
        self.dummies, self.dummy_cols = generate_map_df()

        self.SAVE = SAVE
        self.TIME = TIME
        self.METHOD = METHOD
        self.N_STEPS_AHEAD = N_STEPS_AHEAD
        self.SAVE_PATH = SAVE_PATH


    def run(self):
        """
        The main function to run the encoded LR test.
        """      
        logging.info("thermostat {} - main execution for LR run".format(self.thermostat.tstat_id))

        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))

        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))

            if lr_f.check_if_previously_completed(self.thermostat.tstat_id, test_case):
                self.encoded_dataframe()
                logging.info("thermostat {} - {} generated an encoded dataframe".format(self.thermostat.tstat_id, test_case))
                self.train_test_split_encoded(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))

                if lr_f.test_data_validity_for_train_test(self.test, self.train):
                    self.df_inference_results = self.test.copy()
                    self.df_inference_results = self.df_inference_results.rename(columns={'M_t+0':'M_t'})

                    for n in range(0,self.N_STEPS_AHEAD):
                        train_data = self.data_encoded[self.data_encoded.index.isin(self.train.index)]
                        train_data = train_data.filter(regex="\d_\d_\d|M_t\+{}".format(n))
                        self.train_data_debug = train_data

                        X_train, Y_train = form_dataframe_to_arrays_encoded(train_data)

                        self.generate_LR_model(X_train, Y_train)
                        logging.info("thermostat {} - {} generated LR model - {} step model".format(self.thermostat.tstat_id, test_case, n))

                        self.evaluate_LR_model()
                        if self.SAVE:
                            self.save_train_data(test_case, n, self.SAVE_PATH, self.METHOD)
                            logging.info("thermostat {} - {} saved LR model details - {} step model".format(self.thermostat.tstat_id, test_case, n))

                        self.Test_Data_n_steps_encoded(n)
                        logging.info("thermostat {} - {} extended horizon - {} step model".format(self.thermostat.tstat_id, test_case,n))
                    if self.SAVE:
                        self.save_test_results(test_case, n, self.SAVE_PATH, self.METHOD)
                        logging.info("thermostat {} - saved results - {} step model".format(self.thermostat.tstat_id, test_case, n))

                else:
                    logging.info("thermostat {} - {} invalid data aborting LR test".format(self.thermostat.tstat_id, test_case))
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))

    def encoded_dataframe(self):
        """
        Encode the features in the dataframe 
        """       
        self.data_encoded = self.data.filter(regex="[\+]").copy()
        self.data_encoded['feature'] = self.data.apply(lambda x: str(x['H_t'])+"_"+str(x['W_t'])+"_"+str(x['M_t-1']),axis=1)
        self.data_encoded = self.data_encoded.reset_index().merge(self.dummies,on='feature', how="left").set_index('')

    def train_test_split_encoded(self, split_day):
        """
        Split the dataframe into train and test based on a split day. 
        """      
        self.train, self.test = util.make_train_and_test_dataframes(self.data_encoded, split_day, time_step_buffer=0)

    def Test_Data_n_steps_encoded(self, n_steps=1):
        """
        Peform inference at the nth step for the test and return a data filled dataframe. 
        """        
        for row_i, data in self.test.iterrows():
            if self.TIME:
                tic = time.time()
            pred_ = self.model.predict(data.filter(regex="\d_\d_\d").values.reshape(1,-1))
            if self.TIME:
                toc = time.time()
                print('Elapsed: %s' %(toc-tic))
            self.df_inference_results.at[row_i, 'M_t+{}'.format(n_steps)] = pred_


