from . import lr_framework as lr_f
import os
import numpy as np
import pandas as pd
import itertools
import logging

import json

from . import constants as constant
from . import utils as util

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV


METHOD_NAME = "RF"
SAVE_PATH = os.path.join(os.path.expanduser("~"),"projects/occupancy_state/final_versions",METHOD_NAME)
NSTEPS_AHEAD = 6
SAVE = True
PARAMS = {"max_depth": [3, None], "min_samples_leaf": [1, 3, 10]}

class rf_test(lr_f.lr_test):
    def __init__(self, tstat_id):
        lr_f.lr_test.__init__(self, tstat_id)
        #self.data = self.data.astype('int')

    def generate_RF_model(self, X_train, Y_train):
        # initialize
        #self.model = RandomForestClassifier(n_jobs=2, random_state=0)
        self.model = GridSearchCV(estimator=RandomForestClassifier(),param_grid=PARAMS, n_jobs=-1)
        self.model.fit(X_train,Y_train)

    def evaluate_RF_model(self, feature_order, X_train, Y_train):
        y_pred = self.model.predict(X_train)

        self.model_fit = {}
        self.model_fit['max_depth'] = self.model.best_params_['max_depth']
        self.model_fit['min_samples_leaf'] = self.model.best_params_['min_samples_leaf']
        self.model_fit['accuracy'] = accuracy_score(Y_train,y_pred)
        self.model_fit['precision'] = precision_score(Y_train,y_pred)
        self.model_fit['recall'] = recall_score(Y_train,y_pred)
        self.model_fit['train_accuracy'] = list(zip(feature_order, self.model.best_estimator_.feature_importances_))

    def run(self):
        logging.info("thermostat {} - main execution for {} run".format(self.thermostat.tstat_id, METHOD_NAME))

        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))

        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))

            if lr_f.check_if_previously_completed(self.thermostat.tstat_id, test_case):
                self.train_test_split(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))

                if lr_f.test_data_validity_for_train_test(self.test, self.train):
                    self.df_inference_results = self.test.copy()
                    self.df_inference_results = self.df_inference_results.rename(columns={'M_t+0':'M_t'})
                    for n in range(0,NSTEPS_AHEAD):
                        train_data = self.train.filter(items=['sin_H_t','cos_H_t', 'W_t', 'M_t-1','M_t+{}'.format(n)])
                        self.train_data_debug = train_data
                        X_train, Y_train, feature_order = lr_f.form_dataframe_to_arrays(train_data)
                        self.debug_feature_order = feature_order
                        self.generate_RF_model(X_train, Y_train)
                        logging.info("thermostat {} - {} generated RF model - {} step model".format(self.thermostat.tstat_id, test_case, n))
                        self.evaluate_RF_model(feature_order, X_train, Y_train)
                        if SAVE:
                            self.save_train_data(test_case, n, SAVE_PATH, METHOD_NAME)
                            logging.info("thermostat {} - {} saved RF model details - {} step model".format(self.thermostat.tstat_id, test_case, n))

                        self.Test_Data_n_steps(feature_order,n)
                        logging.info("thermostat {} - {} extended horizon - {} step model".format(self.thermostat.tstat_id, test_case, n))

                    if SAVE:
                        self.save_test_results(test_case, n, SAVE_PATH, METHOD_NAME)
                        logging.info("thermostat {} - saved results - {} step model".format(self.thermostat.tstat_id, test_case, n))

                else:
                    logging.info("thermostat {} - {} invalid data aborting RF test".format(self.thermostat.tstat_id, test_case))
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))


    def save_train_data(self,trial, n, save_path = SAVE_PATH, method=METHOD_NAME):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path,'training_n{}_{}_{}.json'.format(n, self.thermostat.tstat_id, trial)), 'w') as f:
            json.dump(self.model_fit, f)
