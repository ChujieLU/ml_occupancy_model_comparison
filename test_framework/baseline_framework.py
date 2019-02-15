import os
import numpy as np
import pandas as pd
import logging

from . import lr_framework as lr_f
from . import constants as constant
from . import utils as util


#METHOD_NAME = "BASE2_"
#SAVE_PATH = os.path.join(os.path.expanduser("~"),"Documents/PhD_Work/home_occupancy_state/final_versions2",METHOD_NAME)
#NSTEPS_AHEAD = 6
#SAVE = False

def add_prediction_columns(df_, pred_len = 6):
    """
    Add an empty column for each timestep.
    """
    for n_pred in range(pred_len):
        df_['M_t+{}'.format(n_pred)] = np.nan
    return df_

class base_tests(lr_f.lr_test):
    """
    Class for running the three devised baselines
    """
    def __init__(self, tstat_id, METHOD, N_STEPS_AHEAD, SAVE, SAVE_PATH):
        """
        Generall intialization of the class
        """
        lr_f.lr_test.__init__(self, tstat_id)
        self.SAVE = SAVE
        self.METHOD = METHOD
        self.N_STEPS_AHEAD = N_STEPS_AHEAD
        self.SAVE_PATH = SAVE_PATH

    def run(self):
        """
        Main class to run all of the tests for the baselines
        """
        logging.info("thermostat {} - main execution for LR run".format(self.thermostat.tstat_id))

        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))

        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))

            if lr_f.check_if_previously_completed(self.thermostat.tstat_id, test_case):
                self.train_test_split(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))
                if lr_f.test_data_validity_for_train_test(self.test, self.train):
                    self.result_most_common_state()
                    logging.info("thermostat {} - {} Base1 Done".format(self.thermostat.tstat_id, test_case))
                    self.result_most_common_time()
                    logging.info("thermostat {} - {} Base2 Done".format(self.thermostat.tstat_id, test_case))
                    self.result_past_state()
                    logging.info("thermostat {} - {} Base3 Done".format(self.thermostat.tstat_id, test_case))

                    self.save_all_base_results(test_case,self.N_STEPS_AHEAD, self.SAVE_PATH, self.METHOD)


    def result_most_common_state(self):
        """
        Apply the most common state for all the inference times
        """
        self.df_result_frequent_state = self.test.filter(items=['M_t+0']).copy()
        self.df_result_frequent_state = self.df_result_frequent_state.rename(columns={'M_t+0':'M_t'})

        self.df_result_frequent_state = add_prediction_columns(self.df_result_frequent_state, self.N_STEPS_AHEAD)
        max_val = self.most_common_state()

        temp_cols = list(self.df_result_frequent_state.filter(regex='^M_t\+[0-9]$').columns)
        self.df_result_frequent_state[temp_cols] = max_val


    def most_common_state(self):
        """
        Determine the most common state overall
        """
        df_counts = self.train['M_t+0'].dropna().value_counts()
        return df_counts.sort_index().idxmax()


    def result_most_common_time(self):
        """
        Infere values at each prediction based on the most common state in that time bin
        """
        self.df_result_frequent_time = self.test.filter(items=['M_t+0','H_t']).copy()
        self.df_result_frequent_time = self.df_result_frequent_time.rename(columns={'M_t+0':'M_t'})
        self.df_result_frequent_time = self.df_result_frequent_time.reset_index(drop=False)
        max_common_array = self.most_common_by_time_stamp()
        self.debug_HOD = max_common_array
        self.df_result_frequent_time = self.df_result_frequent_time.merge(max_common_array, on='H_t')
        self.df_result_frequent_time = self.df_result_frequent_time.sort_values(['','H_t'])
        self.df_result_frequent_time = self.df_result_frequent_time.reset_index(drop=True)
        for n_pred in range(1,self.N_STEPS_AHEAD):
            self.df_result_frequent_time['M_t+{}'.format(n_pred)] = self.df_result_frequent_time['M_t+0'].shift(-n_pred)


    def most_common_by_time_stamp(self):
        """
        Determine most common state in a time bin
        """
        df_counts = self.train.filter(items=['M_t+0','H_t'])
        df_most_common = df_counts.groupby('H_t').agg(lambda x:x.value_counts().index[0])
        df_most_common = df_most_common.reset_index(drop=False)
        return df_most_common


    def result_past_state(self):
        """
        Generate results using only the past state
        """
        self.df_result_past_state = self.train.filter(items=['M_t+0']).copy()
        self.df_result_past_state = self.df_result_past_state.rename(columns={'M_t+0':'M_t'})
        for n_pred in range(self.N_STEPS_AHEAD):
            self.df_result_past_state['M_t+{}'.format(n_pred)] = self.train['M_t-1']


    def save_all_base_results(self, trial, n, save_path, method):
        """
        Save all of the results
        """
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        self.df_result_frequent_state.to_csv(os.path.join(save_path,"{}_n{}_{}_{}_base1.csv.gz".format(method,n,
                                                self.thermostat.tstat_id,trial)),compression='gzip')
        self.df_result_frequent_time.to_csv(os.path.join(save_path,"{}_n{}_{}_{}_base2.csv.gz".format(method,n,
                                                self.thermostat.tstat_id,trial)),compression='gzip')
        self.df_result_past_state.to_csv(os.path.join(save_path,"{}_n{}_{}_{}_base3.csv.gz".format(method,n,
                                                self.thermostat.tstat_id,trial)),compression='gzip')
