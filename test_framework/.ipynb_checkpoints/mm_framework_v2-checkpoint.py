import pandas as pd
import numpy as np

from . import constants as constant
from . import utils as util
import logging
from . import retrieve_data as rd
import datetime as dt

import json
import os
import time

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD


METHOD_NAME = "MM"
SAVE = True
#SAVE_PATH = os.path.join(os.path.expanduser("~"),"Documents/PhD_Work/home_occupancy_state",METHOD_NAME,"results")
SAVE_PATH = os.path.join(os.path.expanduser("~"),"projects/occupancy_state",METHOD_NAME,"results")
TEST_SPECIFIC_NAME = ''
COLUMN_SEQUENCE = ['M_t-1','H_t','W_t','M_t']
NSTEPS_AHEAD = 6
NSTEPS_BEHIND = 1

def check_if_previously_completed(tstat, test_case):
    if TEST_SPECIFIC_NAME != '':
        if os.path.isfile(os.path.join(SAVE_PATH,"{}_n{}_{}_{}.csv.gz".format(METHOD_NAME, NSTEPS_AHEAD,
                                                tstat,test_case))):
            return False
        else:
            return True
    else:
        if os.path.isfile(os.path.join(SAVE_PATH,"{}_n{}_{}_{}.csv.gz".format(METHOD_NAME,NSTEPS_AHEAD,
                                                tstat,test_case))):
            return False
        else:
            return True

def check_data_in_evidence(evid_dict, dict_unique_vals):
    for k,v in evid_dict.items():
        if v not in dict_unique_vals[k]:
            return False
    return True

def test_data_validity_for_train_test(test_array, train_array):
    for array_ in [test_array, train_array]:
        if array_.shape[0]==0:
            return False
    if train_array['H_t'].value_counts().shape[0]!=48:
        return False
    elif train_array['M_t'].value_counts().shape[0]!=2:
        return False
    return True

def increase_inference_values(tstamp, dict_):
    dict_['H_t'] = dict_['H_t'] + 1
    tstamp = tstamp + dt.timedelta(minutes=30)
    if dict_['H_t'] > 47:
        dict_['W_t'] = util.Is_This_A_Weekday(tstamp)
        dict_['H_t'] = 0
    for t in range(2,NSTEPS_BEHIND+1):
        dict_['M_t-{}'.format(t)] = dict_['M_t-{}'.format(t+1)]
    return tstamp, dict_

def Map_Occ_Values(val):
    """
    Apply a mapping with some logic to make sure mapping is done correctly of
    occupancy states, particularly when NaNs are inccured.
    """
    if np.isnan(val):
        return 4
    elif val >0.5:
        return 1
    else:
        return 0

class mm_test():
    def __init__(self, tstat_id):
        self.thermostat = rd.thermostat_data(tstat_id)
        self.thermostat.main()
        self.data = self.thermostat.df_clean.copy()
        for t in range(1,NSTEPS_BEHIND+1):
            self.data.at[:,'M_t-{}'.format(t)] = self.data['M_t'].shift(t)

        for t in range(1,NSTEPS_AHEAD):
            self.data.at[:,'M_t+{}'.format(t)] = self.data['M_t'].shift(t)
            self.data.at[:,'H_t+{}'.format(t)] = self.data['H_t'].shift(t)
            self.data.at[:,'W_t+{}'.format(t)] = self.data['W_t'].shift(t)

    def prepare_MM_data(self):
        self.data = self.data.dropna()
        self.data.at[:,:] = self.data.astype('int')

    def train_test_split(self, split_day):
        self.train, self.test = util.make_train_and_test_dataframes(self.data, split_day, time_step_buffer=0)

    def generate_MM_model(self):
        """
        This function will initalize and train the markov model with forward sequences

        Input:
        ------

        Output:
        -------

        """
        model_list = []
        for t in range(1,NSTEPS_BEHIND+1):
            model_list.append(('M_t-{}'.format(t), 'M_t'))
        model_list.append(('W_t', 'M_t'))
        model_list.append(('H_t', 'M_t'))

        for t in range(1,NSTEPS_AHEAD):
            if t > 1:
                model_list.append(('M_t+{}'.format(t-1), 'M_t+{}'.format(t)))
            else:
                model_list.append(('M_t'.format(t), 'M_t+{}'.format(t)))
            model_list.append(('W_t+{}'.format(t), 'M_t+{}'.format(t)))
            model_list.append(('H_t+{}'.format(t), 'M_t+{}'.format(t)))


        self.model = BayesianModel(model_list)
        # Learing CPDs using Maximum Likelihood Estimators
        self.model.fit(self.train, estimator=MaximumLikelihoodEstimator)

    def generate_CPD_dictionary(self):
        self.model_table_dicts = {}
        for cpd in self.model.get_cpds():
            self.model_table_dicts[cpd.variable] = cpd

    def Test_Data_Inference(self, df_test):
        """
        Conduct inference on a single day. Function will iterate one time step at a time
        and use all the evidecne from the pre-formed test set.
        """
        df_inference_results = df_test.copy()
        infer = VariableElimination(self.model)

        dict_unique_vals = dict(zip(df_test.columns, [df_test[i].unique() for i in df_test.columns]))

        for key, value in df_test.filter(items=[x for x in df_test.columns if x != 'M_t']).to_dict('index').items():
                value = {k:v for k,v in value.items() if k != 'M_t'}

                if check_data_in_evidence(value, dict_unique_vals):

                    result = infer.query(['M_t'],value)
                    df_inference_results.at[key,'M_t_0'] = result['M_t'].values[0]
                    df_inference_results.at[key,'M_t_1'] = result['M_t'].values[1]

                else:
                    df_inference_results.at[key,'M_t_0'] = np.nan
                    df_inference_results.at[key,'M_t_1'] = np.nan

        return df_inference_results

    def Test_Data_Inference_map_n_steps(self, df_test, n_tsteps):
        # make a function that can predict N timesteps ahead.
        df_inference_results = df_test.filter(items=COLUMN_SEQUENCE).copy()
        infer = VariableElimination(self.model)

        dict_unique_vals = dict(zip(df_test.columns, [df_test[i].unique() for i in df_test.columns]))
        result_list = ['M_t']
        if n_tsteps > 1:
            result_list = result_list+["M_t+{}".format(x) for x in range(1,n_tsteps)]
        count = 0
        for key, value in df_test.filter(items=[x for x in df_test.columns if x not in result_list]).to_dict('index').items():

                index_key = key
                if check_data_in_evidence(value, dict_unique_vals):
                    tic = time.time()
                    result = infer.query(variables=result_list,evidence=value)
                    toc = time.time() - tic
                    logging.info("thermostat {} - Elapsed seconds for query {:.2f}".format(self.thermostat.tstat_id, toc))

                    tic = time.time()
                    map_result = infer.map_query(variables=result_list,evidence=value)
                    toc = time.time() - tic
                    logging.info("thermostat {} - Elapsed seconds for MAP query {:.2f}".format(self.thermostat.tstat_id, toc))

                    for n in result_list:
                        df_inference_results.at[index_key,'{}_0'.format(n)] = result[n].values[0]
                        df_inference_results.at[index_key,'{}_1'.format(n)] = result[n].values[1]
                        df_inference_results.at[index_key,'{}'.format(n)] = Map_Occ_Values(result[n].values[1])
                        df_inference_results.at[index_key, '{}_map'.format(n)] = map_result[n]
                else:
                    for n in result_list:
                        df_inference_results.at[index_key,'{}_0'.format(n)] = np.nan
                        df_inference_results.at[index_key,'{}_1'.format(n)] = np.nan
                        df_inference_results.at[index_key,'{}'.format(n)] = np.nan
                        df_inference_results.at[index_key, '{}_map'.format(n)] = np.nan
                count+=1

        logging.info("thermostat {} - Iterations of test {}".format(self.thermostat.tstat_id, count))
        return df_inference_results

    def Test_Data_Inference_n_steps(self, df_test, n_tsteps):
        # make a function that can predict N timesteps ahead.
        df_inference_results = df_test.filter(items=COLUMN_SEQUENCE).copy()
        infer = VariableElimination(self.model)

        dict_unique_vals = dict(zip(df_test.columns, [df_test[i].unique() for i in df_test.columns]))
        result_list = ['M_t']
        if n_tsteps > 1:
            result_list = result_list+["M_t+{}".format(x) for x in range(1,n_tsteps)]
        count = 0
        self.debug_timmer = []
        for key, value in df_test.filter(items=[x for x in df_test.columns if x not in result_list]).to_dict('index').items():

                index_key = key

                if check_data_in_evidence(value, dict_unique_vals):
                    for query_var in result_list:
                        tic = time.time()
                        result = infer.query(variables=[query_var],evidence=value)
                        toc = time.time() - tic
                        self.debug_timmer.append(toc)
                        df_inference_results.at[index_key,'{}_0'.format(query_var)] = result[query_var].values[0]
                        df_inference_results.at[index_key,'{}_1'.format(query_var)] = result[query_var].values[1]
                        df_inference_results.at[index_key,'{}'.format(query_var)] = Map_Occ_Values(result[query_var].values[1])
                else:
                    for query_var in result_list:
                        df_inference_results.at[index_key,'{}_0'.format(query_var)] = np.nan
                        df_inference_results.at[index_key,'{}_1'.format(query_var)] = np.nan
                        df_inference_results.at[index_key,'{}'.format(query_var)] = np.nan

                count+=1

        logging.info("thermostat {} - Iterations of test {}".format(self.thermostat.tstat_id, count))
        return df_inference_results

    def save_test_results(self,trial):
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        self.horizon_inference_result.to_csv(os.path.join(SAVE_PATH,"{}_n{}_{}_{}.csv.gz".format(METHOD_NAME,NSTEPS_AHEAD,
                                                self.thermostat.tstat_id,trial)),
                                        compression='gzip')

    def run(self,map_method=False):
        logging.info("thermostat {} - main execution for MM run".format(self.thermostat.tstat_id))
        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))
        self.prepare_MM_data()
        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))
            if check_if_previously_completed(self.thermostat.tstat_id, test_case):
                self.train_test_split(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))
                if test_data_validity_for_train_test(self.test, self.train):
                    self.generate_MM_model()
                    logging.info("thermostat {} - {} generated MM model".format(self.thermostat.tstat_id, test_case))

                    self.generate_CPD_dictionary()
                    logging.info("thermostat {} - {} generated CPD dict".format(self.thermostat.tstat_id, test_case))

                    if map_method:
                        self.horizon_inference_result = self.Test_Data_Inference_map_n_steps(self.test, NSTEPS_AHEAD)
                        logging.info("thermostat {} - {} MAP extended horizon".format(self.thermostat.tstat_id, test_case))
                    else:
                        self.horizon_inference_result = self.Test_Data_Inference_n_steps(self.test, NSTEPS_AHEAD)
                        logging.info("thermostat {} - {} extended horizon".format(self.thermostat.tstat_id, test_case))
                    #break
                    if SAVE:
                        self.save_test_results(test_case)
                        logging.info("thermostat {} - saved results".format(self.thermostat.tstat_id, test_case))

                else:
                    logging.info("thermostat {} - {} invalid data aborting MM test".format(self.thermostat.tstat_id, test_case))
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))
            break
