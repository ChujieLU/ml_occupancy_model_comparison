import pandas as pd
import numpy as np
import datetime as dt

from . import constants as constant
from . import utils as util
from . import retrieve_data as rd

import os
import json
import time
import logging

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

C_PARAM_RANGE = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
PARAMS = {'C':C_PARAM_RANGE}

METHOD_NAME = "LR"
SAVE_PATH = os.path.join(os.path.expanduser("~"),"projects/occupancy_state","final_versions",METHOD_NAME)
TEST_SPECIFIC_NAME = ''
COLUMN_SEQUENCE = ['M_t-1','H_t','W_t','M_t']
NSTEPS_AHEAD = 6
SAVE = False

TIME_INFERENCE = False
TIME_TRAINING = True

def form_dataframe_to_arrays(df):
    """
    Convert dataframes into arrays in both variables and target
    """     
    feature_order = list(df.filter(regex="[^M_t+\d]").columns)
    X = df.filter(regex="[^M_t+\d]").values
    Y = df.filter(regex="[\+]").values.flatten()
    return X,Y,feature_order

def check_if_previously_completed(tstat, test_case):
    """ 
    This currently is not functional because of file name changes
    """
    if TEST_SPECIFIC_NAME != '':
        if os.path.isfile(os.path.join(SAVE_PATH,"{}_n{}_{}_{}.csv.gz".format(METHOD_NAME,NSTEPS_AHEAD,
                                                tstat,test_case))):
            return False
        else:
            return True
    else:
        if os.path.isfile(os.path.join(SAVE_PATH,'training_{}_{}_{}_{}.json'.format(METHOD_NAME,tstat,test_case,TEST_SPECIFIC_NAME))):
            return False
        else:
            return True

def test_data_validity_for_train_test(test_array, train_array):
    """ 
    Check that there is still some data after splitting and cleaning
    """
    for array_ in [test_array, train_array]:
        if array_.shape[0]==0:
            return False
    if train_array['M_t+0'].value_counts().shape[0]!=2:
        return False
    return True


class lr_test(object):
    """ 
    Class for non-encoded LR. It is imported by many other functions. 
    """
    def __init__(self, tstat_id):
        """ 
        Initalization of class
        """
        self.thermostat = rd.thermostat_data(tstat_id)
        self.thermostat.main()
        self.data = self.thermostat.df_clean.copy()
        self.data.at[:,'M_t-1'] = self.data['M_t'].shift(+1)
        for n in range(1,NSTEPS_AHEAD):
            self.data.at[:,'M_t+{}'.format(n)] = self.data['M_t'].shift(-1*n)
        self.data = self.data.rename(columns={'M_t':'M_t+0'})
        self.data['sin_H_t'] = np.sin(2*np.pi*self.data['H_t']/48)
        self.data['cos_H_t'] = np.cos(2*np.pi*self.data['H_t']/48)
        self.data = self.data.dropna(axis=0)

    def run(self):
        logging.info("thermostat {} - main execution for LR run".format(self.thermostat.tstat_id))
        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))
        for test_case in ['test_1','test_2','test_3','test_4']:

            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))

            if check_if_previously_completed(self.thermostat.tstat_id, test_case):
                self.train_test_split(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))
                if test_data_validity_for_train_test(self.test, self.train):
                    self.df_inference_results = self.test.copy()
                    self.df_inference_results = self.df_inference_results.rename(columns={'M_t+0':'M_t'})
                    for n in range(0,NSTEPS_AHEAD):
                        train_data = self.train.filter(items=['sin_H_t','cos_H_t', 'W_t', 'M_t-1','M_t+{}'.format(n)])
                        self.train_data_debug = train_data
                        X_train, Y_train, feature_order = form_dataframe_to_arrays(train_data)
                        self.generate_LR_model(X_train, Y_train)
                        logging.info("thermostat {} - {} generated LR model - {} step model".format(self.thermostat.tstat_id, test_case, n))

                        self.evaluate_LR_model()
                        if SAVE:
                            self.save_train_data(test_case, n)
                            logging.info("thermostat {} - {} saved LR model details - {} step model".format(self.thermostat.tstat_id, test_case, n))

                        #self.horizon_inference_result = self.Test_Data_n_steps(NSTEPS_AHEAD)
                        self.Test_Data_n_steps(feature_order, n)
                        logging.info("thermostat {} - {} extended horizon - {} step model".format(self.thermostat.tstat_id, test_case, n))
                    if SAVE:
                        self.save_test_results(test_case, n)
                        logging.info("thermostat {} - saved results - {} step model".format(self.thermostat.tstat_id, test_case, n))

                else:
                    logging.info("thermostat {} - {} invalid data aborting LR test".format(self.thermostat.tstat_id, test_case))
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))
            #break

    def train_test_split(self, split_day):
        """ 
        Split dataframe to a train and test set
        """
        self.train, self.test = util.make_train_and_test_dataframes(self.data, split_day, time_step_buffer=0)

    def generate_LR_model(self, X_train, Y_train):
        """ 
        Initailze and fit the LR module
        """
        if TIME_TRAINING:
            tic = time.time()
        self.model = GridSearchCV(estimator=linear_model.LogisticRegression(), param_grid=PARAMS, n_jobs=-1)
        self.model.fit(X_train,Y_train)
        if TIME_TRAINING:
            toc = time.time()
            print('Elapsed time training: %s' %(toc-tic))

    def evaluate_LR_model(self):
        """ 
        determine best scores after cross validation
        """
        self.LR_model_fit = {}
        self.LR_model_fit['C_value'] = self.model.best_params_['C']
        self.LR_model_fit['train_accuracy'] = self.model.best_score_

    def save_train_data(self,trial,n,save_path = SAVE_PATH, method=METHOD_NAME):
        """ 
        Save the the training data parameters for each trail
        """
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path,'training_n{}_{}_{}.json'.format(n, self.thermostat.tstat_id, trial)), 'w') as f:
            json.dump(self.LR_model_fit, f)

    def Test_Data_n_steps(self, feature_order, n_steps=1):
        """ 
        Conduct inference at specific time horizon for all timesteps.
        """
        for row_i, data in self.test.iterrows():
            if TIME_INFERENCE:
                tic = time.time()
            pred_ = self.model.predict(data.filter(feature_order).values.reshape(1,-1))
            if TIME_INFERENCE:
                toc = time.time()
                print('Elapsed time inference: %s' %(toc-tic))
            self.df_inference_results.at[row_i, 'M_t+{}'.format(n_steps)] = pred_


    def save_test_results(self,trial,n, save_path = SAVE_PATH, method=METHOD_NAME):
        """ 
        Save test result dataframe to a csv.
        """
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        self.df_inference_results.to_csv(os.path.join(save_path,"{}_n{}_{}_{}.csv.gz".format(method,n,
                                                self.thermostat.tstat_id,trial)),
                                        compression='gzip')

