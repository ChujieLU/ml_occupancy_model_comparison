import pandas as pd
import numpy as np
import logging
import datetime as dt

from . import constants as constant
from . import utils as util
from . import retrieve_data as rd

import os
import time

from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD


TEST_SPECIFIC_NAME = ''

COLUMN_SEQUENCE = ['M_t-1','H_t','W_t','M_t']
TIME_INFERENCE = False
TIME_TRAINING = True

def check_if_previously_completed(tstat, test_case, SAVE_PATH, METHOD_NAME, NSTEPS_AHEAD):
    """
    Check if work has already been done. 
    """
    if TEST_SPECIFIC_NAME != '':
        if os.path.isfile(os.path.join(SAVE_PATH,"{}_n1{}_{}_{}.csv.gz".format(METHOD_NAME, NSTEPS_AHEAD,
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
    """
    Ensure variables you will condition on existed during traning. 
    """
    for k,v in evid_dict.items():
        if v not in dict_unique_vals[k]:
            return False
    return True

def test_data_validity_for_train_test(test_array, train_array):
    """
    Check that the train data is valid. 
    """
    for array_ in [test_array, train_array]:
        if array_.shape[0]==0:
            return False
    if train_array['H_t'].value_counts().shape[0]!=48:
        return False
    elif train_array['M_t'].value_counts().shape[0]!=2:
        return False
    return True

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
    """
    Class to perform the Markov Model Test. 
    """
    def __init__(self, tstat_id, METHOD, N_STEPS_AHEAD, N_STEPS_BEHIND, SAVE, SAVE_PATH):
        """
        Initialize the class and form the data correctly based on chain size specifications.
        """
        self.thermostat = rd.thermostat_data(tstat_id)
        self.thermostat.main()
        self.data = self.thermostat.df_clean.copy()

        self.SAVE = SAVE
        self.METHOD = METHOD
        self.N_STEPS_AHEAD = N_STEPS_AHEAD
        self.N_STEPS_BEHIND = N_STEPS_BEHIND
        self.SAVE_PATH = SAVE_PATH

        for t in range(1,self.N_STEPS_BEHIND+1):
            self.data.at[:,'M_t-{}'.format(t)] = self.data['M_t'].shift(t)

        for t in range(1,self.N_STEPS_AHEAD):
            self.data.at[:,'M_t+{}'.format(t)] = self.data['M_t'].shift(-t)
            self.data.at[:,'H_t+{}'.format(t)] = self.data['H_t'].shift(-t)
            self.data.at[:,'W_t+{}'.format(t)] = self.data['W_t'].shift(-t)
    
    def run(self,map_method=False):
        """
        Run the test. 
        """
        logging.info("thermostat {} - main execution for MM run".format(self.thermostat.tstat_id))
        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))
        self.prepare_MM_data()
        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))
            if check_if_previously_completed(self.thermostat.tstat_id, test_case, self.SAVE_PATH, self.METHOD, self.N_STEPS_AHEAD):
                self.train_test_split(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))
                if test_data_validity_for_train_test(self.test, self.train):
                    self.generate_MM_model()
                    logging.info("thermostat {} - {} generated MM model".format(self.thermostat.tstat_id, test_case))

                    self.generate_CPD_dictionary()
                    logging.info("thermostat {} - {} generated CPD dict".format(self.thermostat.tstat_id, test_case))

                    if map_method:
                        self.horizon_inference_result = self.Test_Data_Inference_map_n_steps(self.test, self.N_STEPS_AHEAD)
                        logging.info("thermostat {} - {} MAP extended horizon".format(self.thermostat.tstat_id, test_case))
                    else:
                        self.horizon_inference_result = self.Test_Data_Inference_n_steps(self.test, self.N_STEPS_AHEAD)
                        logging.info("thermostat {} - {} extended horizon".format(self.thermostat.tstat_id, test_case))
                    
                    if self.SAVE:
                        self.save_test_results(test_case)
                        logging.info("thermostat {} - saved results".format(self.thermostat.tstat_id, test_case))

                else:
                    logging.info("thermostat {} - {} invalid data aborting MM test".format(self.thermostat.tstat_id, test_case))
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))

    def prepare_MM_data(self):
        """
        Do final cleaning and type conversion. Pgmpy likes integers. 
        """
        self.data = self.data.dropna()
        self.data.at[:,:] = self.data.astype('int')

    def train_test_split(self, split_day):
        """
        Perform the train/test split based on split day. 
        """
        self.train, self.test = util.make_train_and_test_dataframes(self.data, split_day, time_step_buffer=0)


    def generate_MM_model(self):
        """
        This function will initalize and train the markov model with forward sequences.
        """
        model_list = []
        for t in range(1,self.N_STEPS_BEHIND+1):
            model_list.append(('M_t-{}'.format(t), 'M_t'))
        model_list.append(('W_t', 'M_t'))
        model_list.append(('H_t', 'M_t'))

        for t in range(1,self.N_STEPS_AHEAD):
            if t > 1:
                model_list.append(('M_t+{}'.format(t-1), 'M_t+{}'.format(t)))
            else:
                model_list.append(('M_t'.format(t), 'M_t+{}'.format(t)))
            model_list.append(('W_t+{}'.format(t), 'M_t+{}'.format(t)))
            model_list.append(('H_t+{}'.format(t), 'M_t+{}'.format(t)))

        if TIME_TRAINING:
            tic = time.time()
        self.model = BayesianModel(model_list)
        # Learing CPDs using Maximum Likelihood Estimators
        self.model.fit(self.train, estimator=MaximumLikelihoodEstimator)
        if TIME_TRAINING:
            toc = time.time()
            print('Elapsed time training: %s' %(toc-tic))

    def generate_CPD_dictionary(self):
        """
        Place cpd in dictionary to be accessible later/in debug. 
        """
        self.model_table_dicts = {}
        for cpd in self.model.get_cpds():
            self.model_table_dicts[cpd.variable] = cpd

    def Test_Data_Inference_map_n_steps(self, df_test, n_tsteps):
        """
        Perform both map and marignal inference and report values
        """
        df_inference_results = df_test.filter(items=COLUMN_SEQUENCE).copy()
        df_inference_results['M_t_orig'] = df_inference_results['M_t']
        infer = VariableElimination(self.model)

        dict_unique_vals = dict(zip(df_test.columns, [df_test[i].unique() for i in df_test.columns]))
        result_list = ['M_t']
        if n_tsteps > 1:
            result_list = result_list+["M_t+{}".format(x) for x in range(1,n_tsteps)]
        count = 0
        for key, value in df_test.filter(items=[x for x in df_test.columns if x not in result_list]).to_dict('index').items():

                index_key = key
                if check_data_in_evidence(value, dict_unique_vals):

                    #MAP query
                    tic = time.time()
                    map_result = infer.map_query(variables=result_list,evidence=value)
                    toc = time.time() - tic
                    logging.info("thermostat {} - Elapsed seconds for MAP query {:.2f}".format(self.thermostat.tstat_id, toc))

                    for n in result_list:
                        
                        tic = time.time()
                        result = infer.query(variables=[n], evidence=value)
                        
                        toc = time.time()
                        if TIME_INFERENCE:
                            print('Elapsed: %s' %(toc-tic))
                        logging.info("thermostat {} - Elapsed seconds for query {:.2f}".format(self.thermostat.tstat_id, toc))

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
        """
        Perform maginal inference and return values. 
        """
        df_inference_results = df_test.filter(items=COLUMN_SEQUENCE).copy()
        df_inference_results['M_t_orig'] = df_inference_results['M_t']
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
        """
        Save the results of the test to csv.
        """    
        if not os.path.isdir(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)

        self.horizon_inference_result.to_csv(os.path.join(self.SAVE_PATH,"{}_n{}_{}_{}.csv.gz".format(self.METHOD,self.N_STEPS_AHEAD,
                                                self.thermostat.tstat_id,trial)),
                                        compression='gzip')
