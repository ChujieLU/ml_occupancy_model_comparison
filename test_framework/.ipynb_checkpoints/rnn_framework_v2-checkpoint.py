import pandas as pd
import numpy as np
from . import constants as constant
from . import utils as util
import logging
from . import retrieve_data as rd
import json
import os
from . import lr_framework as lr_f

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Activation, TimeDistributed, Dense, Bidirectional, Dropout, LSTM
from tensorflow.python.keras.callbacks import Callback

INPUT_LAYERS = 1
HIDDEN_LAYER_SIZE = 8
DROPOUT = 0.2
COLUMN_NAMES = ['sin_H_t','cos_H_t','W_t','M_t-1']
RNN_COLUMNS =  [ 'W_t', 'sin_H_t', 'cos_H_t']

METHOD_NAME = "RNN"

SAVE_PATH = os.path.join(os.path.expanduser("~"),"Documents/PhD_Work/home_occupancy_state",METHOD_NAME,"results")

N_BATCH = 4
N_EPOCH = 20
N_LOOKBACK = 24 #12 hours
N_LOOKFORWARD = 6 #3 hours

SAVE = False

def check_if_previously_completed(tstat, test_case):
    #TEST_SPECIFIC_NAME
    if os.path.isfile(os.path.join(SAVE_PATH,'training_{}_{}.json'.format(tstat,test_case))):
        return False
    else:
        return True

def one_hot_occupancy(col_, val):
    if int(col_[-1]) == val:
        return 1
    else:
        return 0

def test_data_validity_for_train_test(test_Xarray, test_Yarray, train_Xarray, train_Yarray):
    for array_ in [test_Xarray, test_Yarray, train_Xarray, train_Yarray]:
        if array_.shape[0]==0:
            return False
    return True

def form_arrays(encode, time_step_buffer=12, time_step_ahead = 1):
    X_COLUMNS = RNN_COLUMNS
    size_ = time_step_buffer+time_step_ahead
    row = 0
    index_list = []
    X_, Y_ = np.array([]), np.array([])
    for ind_ in encode.index[time_step_buffer+time_step_ahead:]:
        df_sample = encode[ind_-pd.Timedelta('0.5H')*(size_-1):ind_].copy()

        if df_sample.shape[0] == size_:
            index_list.append(df_sample.index[-1])

            if time_step_ahead-1 > 0:
                df_sample.at[df_sample.index[-(time_step_ahead-1)]:,'M_t-1_unobs'] = 1
                df_sample.at[df_sample.index[-(time_step_ahead-1)]:,'M_t-1_0'] = 0
                df_sample.at[df_sample.index[-(time_step_ahead-1)]:,'M_t-1_1'] = 0
            if row == 0:
                X_ = df_sample.as_matrix(columns=X_COLUMNS).reshape(1,size_,len(X_COLUMNS))
                Y_ = df_sample.as_matrix(columns=['M_t']).reshape(1,size_,1)
            else:
                new_slice = df_sample.as_matrix(columns=X_COLUMNS).reshape(1,size_,len(X_COLUMNS))
                X_ = np.vstack((X_,new_slice))
                Y_ = np.vstack((Y_, df_sample.as_matrix(columns=['M_t']).reshape(1,size_,1)))

            row += 1

    return X_, Y_, index_list

def generate_uni_directional_RNN_model():
    tf.reset_default_graph()
    try:
        del model
    except:
        pass
    K.clear_session()

    FEATURES = self.X_test.shape[2]
    LOOKBACK = self.X_test.shape[1]

    model = Sequential()

    for layer_number in range(INPUT_LAYERS):
        model.add(LSTM(HIDDEN_LAYER_SIZE, input_shape=(LOOKBACK, FEATURES), kernel_initializer="he_normal",
                                 return_sequences=True))
        model.add(Dropout(DROPOUT = 0.2))
    model.add(TimeDistributed(Dense(1, kernel_initializer="he_normal")))

    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

class rnn_test(lr_f.lr_test):
    def __init__(self, tstat_id):
        lr_f.lr_test.__init__(self, tstat_id)
        self.data = self.data.rename(columns={'M_t+0':'M_t'})

    def train_RNN(self, model):
        num_epoch = N_EPOCH
        n_batch_size = N_BATCH
        history = model.fit(self.X_train, self.Y_train, batch_size=n_batch_size,
                            epochs=num_epoch, validation_data=(self.X_test, self.Y_test))
        self.train_accuracy =  history.history['acc']
        self.test_accuarcy = history.history['val_acc']
        return history

    def generate_model(self):
        self.rnn_model = generate_uni_directional_RNN_model()

    def encode_values(self):
        self.encoded = self.data.filter(items=['M_t-1','M_t','W_t','sin_H_t','cos_H_t'])

        for col_ in ['M_t-1_0','M_t-1_1']:
            self.encoded[col_] = self.data.filter(items=['M_t-1']).applymap(lambda x: one_hot_occupancy(col_, x))
        self.encoded['M_t-1_unobs'] = 0

    def form_train_test_to_arrays(self):
        self.X_train, self.Y_train, self.train_index = form_arrays(self.train, N_LOOKBACK, N_LOOKFORWARD)
        self.X_test, self.Y_test, self.test_index = form_arrays(self.test, N_LOOKBACK, N_LOOKFORWARD)

    def save_test_results(self,trial):
        if not os.path.isdir(SAVE_PATH):
            os.path.mkdir(SAVE_PATH)
        self.df_simulated_result.to_csv(os.path.join(SAVE_PATH,"{}_{}_{}.csv.gz".format(METHOD_NAME,self.thermostat.tstat_id,trial)),
                                        compression='gzip')


    def save_train_data(self,trial,history):
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        training_eval = {}
        training_eval['train'] = history.history['acc']
        training_eval['train'] = history.history['val_acc']

        with open(os.path.join(SAVE_PATH,'training_{}_{}.json'.format(self.thermostat.tstat_id,trial)), 'w') as f:
            json.dump(training_eval, f)

    def populate_prediction_comparison(self, model):
        #TODO confirm this
        self.df_simulated_result = pd.DataFrame(index = self.test_index, columns = ["real","predict","error"])
        for ind in range(self.X_test.shape[0]):
            df_result = pd.DataFrame(columns = ["real","predict"], index=range(self.X_test.shape[1]))
            rowX, rowy = self.X_test[np.array([ind])], self.Y_test[np.array([ind])]

            preds = model.predict_classes(rowX, verbose=0)
            df_result['real'] = rowy.flatten()
            df_result['predict'] = preds.flatten()
            df_result['error'] = df_result.apply(lambda row: util.map_error_fnc(row['real'], row['predict']), axis=1)
            self.df_simulated_result.iat[ind,0] = df_result.iloc[-1,0]
            self.df_simulated_result.iat[ind,1] = df_result.iloc[-1,1]
            self.df_simulated_result.iat[ind,2] = df_result.iloc[-1,2]


    def run(self):
        logging.info("thermostat {} - main execution for RNN run".format(self.thermostat.tstat_id))
        self.encode_values()
        test_days = util.import_train_days()
        logging.info("thermostat {} - load train days".format(self.thermostat.tstat_id))
        for test_case in ['test_1','test_2','test_3','test_4']:
            logging.info("thermostat {} - beginning {}".format(self.thermostat.tstat_id, test_case))
            if check_if_previously_completed(self.thermostat.tstat_id, test_case):
                self.train_test_split(test_days[test_case])
                logging.info("thermostat {} - {} train/test".format(self.thermostat.tstat_id, test_case))
                self.form_train_test_to_arrays()

                if test_data_validity_for_train_test(self.X_test, self.Y_test, self.X_train, self.Y_train):
                    logging.info("thermostat {} - {} train/test array coded".format(self.thermostat.tstat_id, test_case))
                    logging.info("thermostat {} - {} train shape X:{} Y:{}".format(self.thermostat.tstat_id, test_case,
                                self.X_train.shape, self.Y_train.shape))
                    logging.info("thermostat {} - {} test shape X:{} Y:{}".format(self.thermostat.tstat_id, test_case,
                                self.X_test.shape, self.Y_test.shape))

                    model = self.generate_uni_directional_RNN_model()
                    logging.info("thermostat {} - {} generate model".format(self.thermostat.tstat_id, test_case))

                    history = self.train_RNN(model)
                    if SAVE:
                        self.save_train_data(test_case,history)
                        logging.info("thermostat {} - {} saved training info".format(self.thermostat.tstat_id, test_case))

                    self.populate_prediction_comparison(model)
                    logging.info("thermostat {} - ran test".format(self.thermostat.tstat_id, test_case))
                    if SAVE:
                        self.save_test_results(test_case)
                        logging.info("thermostat {} - saved results".format(self.thermostat.tstat_id, test_case))
                else:
                    logging.info("thermostat {} - {} invalid data aborting test".format(self.thermostat.tstat_id, test_case))
            else:
                logging.info("thermostat {} - {} Hey look we already did this one!".format(self.thermostat.tstat_id, test_case))
