import pandas as pd
import numpy as np
import logging

import json
import os

import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

from . import lr_framework as lr_f
from . import retrieve_data as rd
from . import constants as constant
from . import utils as util

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Activation, TimeDistributed, Dense, Bidirectional, Dropout, LSTM
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.optimizers import Adam


N_LOOKBACK = 24 #12 hours
N_LOOKFORWARD = 6 #3 hours
FEATURE_SIZE = 3*48*2

TIME_INFERENCE = False
TIME_TRAINING = True

def check_if_previously_completed(tstat, test_case, SAVE_PATH):
    """
    Check if results from training had been completed.
    """
    if os.path.isfile(os.path.join(SAVE_PATH,'training_{}_{}.json'.format(tstat,test_case))):
        return False
    else:
        return True
    
def show_curve(history):
    """
    Show the performance curves of epochs from traning the RNN. 
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def hash_value(M1, H, W, n_M1 = 3, n_H = 48, n_W = 2):
    """
    Function to generate an encoded value
    """
    return W + n_W*H + n_H*n_W*M1 
    
def test_data_validity_for_train_test(test_Xarray, test_Yarray, train_Xarray, train_Yarray):
    """
    Test for data existing in all train/test sets
    """
    for array_ in [test_Xarray, test_Yarray, train_Xarray, train_Yarray]:
        if array_.shape[0]==0:
            return False
    return True


def build_enocde_array(n_tsteps, n_features):
    """
    Initialize 0-filled array 
    """   
    return np.zeros((n_tsteps, n_features))


def form_arrays(encode, time_step_buffer=12, time_step_ahead = 1):
    """
    Build large input arrays for use with the RNN
    """
    size_ = time_step_buffer+time_step_ahead
    index_list = []

    X_array = np.array([])
    Y_array = np.array([])
    pd_index = pd.Index([])
    
    x_list = []
    y_list = []
    
    if encode.shape[0]>0:
        new_index = pd.DatetimeIndex(start=encode.index[0],
                                end=encode.index[-1],freq='30min')

        new_rnn_encoded = pd.DataFrame(index=new_index)
        new_rnn_encoded = new_rnn_encoded.join(encode)

        rnn_e_v = new_rnn_encoded.values

        tic = time.time()

        for ind_ in range(N_LOOKBACK, rnn_e_v.shape[0]-N_LOOKFORWARD):

            sample = rnn_e_v[ind_-N_LOOKBACK:ind_+(N_LOOKFORWARD),:]
            sample = sample[~np.isnan(sample).all(axis=1)]

            if sample.shape[0] == size_:
                index_list.append(new_rnn_encoded.index[ind_])
                x_array = build_enocde_array(size_, FEATURE_SIZE)

                sample[-N_LOOKFORWARD:,-1] = hash_value(2, sample[-N_LOOKFORWARD:,1], sample[-N_LOOKFORWARD:,2])

                x_array[np.arange(0,size_),sample[:,-1].astype(int)] = 1
                y_array = sample[:,0].reshape(1,size_,-1)
                x_array = x_array.reshape(1,size_,-1)

                x_list.append(x_array)
                y_list.append(y_array)

        X_array = np.vstack(x_list)
        Y_array = np.vstack(y_list)

        toc = time.time() - tic
        logging.info("Elapsed seconds for building arrays {:.2f}".format(toc))

        if len(index_list)>0:
            pd_index = pd.Index(index_list).drop_duplicates()
        
    return X_array, Y_array, pd_index

def error_test(true, predict):
    """
    Function for classifcation of errors.
    """
    if true == predict:
        return 1
    else:
        return 0

class rnn_test(lr_f.lr_test):
    """
    Class for conducting the RNN test
    """
    def __init__(self, tstat_id, save_path, method_name, n_batch, n_epoch,
                input_layers, hidden_layer_size, dropout, save, plot):
        """
        Initialization including many hyper parameters. 
        """
        lr_f.lr_test.__init__(self, tstat_id)
        self.data = self.data.rename(columns={'M_t+0':'M_t'})
        self.SAVE_PATH = save_path
        self.METHOD_NAME = method_name
        self.N_BATCH = n_batch
        self.N_EPOCH = n_epoch
        self.INPUT_LAYERS = input_layers
        self.HIDDEN_LAYER_SIZE = hidden_layer_size
        self.DROPOUT = dropout
        self.SAVE = save
        self.PLOT = plot

        
    def run(self):
        """
        Main function for running tests. 
        """
        logging.info("thermostat {} - main execution for RNN run".format(self.thermostat.tstat_id))
        self.encode_values()
        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))
        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))
            if check_if_previously_completed(self.thermostat.tstat_id, test_case, self.SAVE_PATH):

                self.train_test_split_RNN(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))
                self.form_train_test_to_arrays()

                if test_data_validity_for_train_test(self.X_test, self.Y_test, self.X_train, self.Y_train):
                    logging.info("thermostat {} - {} train/test array coded".format(self.thermostat.tstat_id, test_case))
                    logging.info("thermostat {} - {} train shape X:{} Y:{}".format(self.thermostat.tstat_id, test_case,
                                self.X_train.shape, self.Y_train.shape))
                    logging.info("thermostat {} - {} test shape X:{} Y:{}".format(self.thermostat.tstat_id, test_case,
                                self.X_test.shape, self.Y_test.shape))

                    self.generate_model()
                    logging.info("thermostat {} - {} generate model".format(self.thermostat.tstat_id, test_case))

                    history = self.train_RNN(self.rnn_model)
                    
                    if self.PLOT:
                        show_curve(history)
                    
                    if self.SAVE:
                        self.save_train_data(test_case,history)
                        logging.info("thermostat {} - {} saved training info".format(self.thermostat.tstat_id, test_case))

                    self.populate_prediction_comparison(self.rnn_model, N_LOOKFORWARD)
                    logging.info("thermostat {} - ran test".format(self.thermostat.tstat_id, test_case))
                    if self.SAVE:
                        self.save_test_results(test_case)
                        logging.info("thermostat {} - saved results".format(self.thermostat.tstat_id, test_case))
                else:
                    logging.info("thermostat {} - {} invalid data aborting test".format(self.thermostat.tstat_id, test_case))
                    logging.info("Got here")
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))
                self.load_test_results(test_case)

    def encode_values(self):
        """
        Generate an enoded value based on the combination of features.
        """        
        self.encoded = self.data.filter(items=['M_t','H_t','W_t','M_t-1'])
        self.encoded['hash'] = self.encoded.apply(lambda x: hash_value(x['M_t-1'],x['H_t'],x['W_t']),axis=1)
    
    def train_test_split_RNN(self, split_day):
        """
        Split the data into train and test dataframes
        """     
        self.train, self.test = util.make_train_and_test_dataframes(self.encoded, split_day, time_step_buffer=N_LOOKBACK)

    def form_train_test_to_arrays(self):
        """
        Convert train and test dataframes to arrays
        """
        self.X_train, self.Y_train, self.train_index = form_arrays(self.train, N_LOOKBACK, N_LOOKFORWARD)
        self.X_test, self.Y_test, self.test_index = form_arrays(self.test, N_LOOKBACK, N_LOOKFORWARD)

    
    def generate_model(self):
        """
        Generate and return a model. Could be changed to different architectures.
        """
        self.rnn_model = self.generate_uni_directional_RNN_model()

    def generate_uni_directional_RNN_model(self):
        """
        Initalize a model in keras. Here only a uni-directional RNN. 
        """
        tf.reset_default_graph()
        try:
            del model
        except:
            pass
        K.clear_session()

        FEATURES = self.X_test.shape[2]
        LOOKBACK = self.X_test.shape[1]

        model = Sequential()

        for layer_number in range(self.INPUT_LAYERS):
            model.add(LSTM(self.HIDDEN_LAYER_SIZE, input_shape=(LOOKBACK, FEATURES), kernel_initializer="he_normal",
                                     return_sequences=True))
            model.add(Dropout(self.DROPOUT))
        model.add(TimeDistributed(Dense(1, kernel_initializer="he_normal")))

        model.add(Activation('sigmoid'))
        adm = Adam(lr=0.001) #default lr = 0.001 ep1e-8
        model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['acc'])
        return model

    def train_RNN(self, model):
        """
        Provided a moel, run the training given a number of hyperparameters. 
        """
        if TIME_TRAINING:
            tic = time.time()
        num_epoch = self.N_EPOCH
        if self.N_BATCH == None:
            n_batch_size = self.X_train.shape[0]
        else:
            n_batch_size = self.N_BATCH
        history = model.fit(self.X_train, self.Y_train, batch_size=n_batch_size,
                            epochs=num_epoch, validation_data=(self.X_test, self.Y_test))
        self.train_accuracy =  history.history['acc']
        self.test_accuarcy = history.history['val_acc']
        if TIME_TRAINING:
            toc = time.time()
            print('Elapsed time training: %s' %(toc-tic))       
        return history

    def save_train_data(self,trial,history):
        """
        Save the results of training into a json object.
        """
        if not os.path.isdir(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)
        training_eval = {}
        training_eval['train'] = history.history['acc']
        training_eval['train'] = history.history['val_acc']

        with open(os.path.join(self.SAVE_PATH,'training_{}_{}.json'.format(self.thermostat.tstat_id,trial)), 'w') as f:
            json.dump(training_eval, f)

    def populate_prediction_comparison(self, model, n_steps):
        """
        Conduct inference on the test set using the trained model.
        """
        self.df_simulated_result = pd.DataFrame(index = self.test_index)
        self.df_simulated_result = self.df_simulated_result.join(self.test.filter(items=['M_t']))
        for n in range(0,n_steps):
            self.df_simulated_result['M_t+{}'.format(n)] = np.nan

        for ind in range(self.X_test.shape[0]):
            df_result = pd.DataFrame(columns = ["real","predict"], index=range(self.X_test.shape[1]))
            rowX, rowy = self.X_test[np.array([ind])], self.Y_test[np.array([ind])]
            tic = time.time()
            preds = model.predict_classes(rowX, verbose=0)
            toc = time.time()
            if TIME_INFERENCE:
                print('Elapsed: %s' %(toc-tic))

            df_result['real'] = rowy.flatten()
            df_result['predict'] = preds.flatten()
            
            for n in range(0,n_steps):
                self.df_simulated_result.iat[ind,1*(n+1)] = df_result.iloc[-1*(n_steps - n),1]


    def save_test_results(self,trial):
        """
        Save results of inference on test set to a csv.
        """
        if not os.path.isdir(self.SAVE_PATH):
            os.path.mkdir(self.SAVE_PATH)
        self.df_simulated_result.to_csv(os.path.join(self.SAVE_PATH,"{}_{}_{}.csv.gz".format(self.METHOD_NAME,self.thermostat.tstat_id,trial)),
                                        compression='gzip')

    def load_test_results(self,trial):
        """
        Load in previous results if restart had to happen.
        """
        self.df_simulated_result = pd.read_csv(os.path.join(self.SAVE_PATH,"{}_{}_{}.csv.gz".format(self.METHOD_NAME,self.thermostat.tstat_id,trial)),
                                        compression='gzip')

        self.df_simulated_result['Unnamed: 0'] = pd.to_datetime(self.df_simulated_result.loc[:,'Unnamed: 0'])
        self.df_simulated_result = self.df_simulated_result.set_index(['Unnamed: 0'],drop=True)


    def tunning_average_daily_accuracy(self):
        """
        Useful function for helping with hyperparamter tunning.
        """
        try:
            test_ = self.df_simulated_result.copy()
            test_['test_situation'] = test_.apply(lambda x: error_test(x['M_t'],x['M_t+0']),axis=1)
            test_ = test_.groupby(pd.Grouper(freq="D")).agg({'test_situation':'mean'})
            return test_.mean()[0]
        except AttributeError:
            return np.nan
        
        
    def tunning_average_daily_accuracy_at_steps(self, TSTEPS=6, DAILY_DATA_THRES=24):
        """
        Useful function for helping with reporting by returning dictionary of accuracy values
        """
        acc_dict = {}

        for n in range(0,TSTEPS):
            acc_dict['M_t+{}'.format(n)] = np.nan
            
            try:
                df_single_case = self.df_simulated_result.filter(regex='^M_t\+{}$|^M_t$'.format(n))
                df_single_case.at[:,'M_t'] = df_single_case.loc[:,'M_t'].shift(-n)
                df_single_case = df_single_case.dropna()

                accuracy_list = []

                for i, (date, day_data) in enumerate(df_single_case.groupby(pd.Grouper(freq='D'))):
                    if day_data.shape[0] > DAILY_DATA_THRES:

                        pred_ = list(day_data['M_t+{}'.format(n)])
                        real_ = list(day_data['M_t'])

                        acc_ = accuracy_score(real_,pred_)

                        accuracy_list.append(acc_)

                acc_dict['M_t+{}'.format(n)] = np.mean(accuracy_list)
            except AttributeError:
                pass
            
        return acc_dict
