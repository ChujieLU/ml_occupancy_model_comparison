import os
import pandas as pd
import numpy as np
from . import constants as constant
from operator import add
import json

def dataload_n_clean(ident):
    """
    Read in a compressed file and return a dataframe.
    
    Inputs:
    -------
    ident -> the name of the thermostat

    Outputs:
    --------
    df_ -> the loaded and cleaned dataframe
    """
    tstat_path = os.path.join(constant.RAW_DATA_DIRECTORY,'{}.xz'.format(ident))
    df_ = pd.read_csv(tstat_path , compression='xz')
    df_['Unnamed: 0'] = pd.to_datetime(df_['Unnamed: 0'])
    df_ = df_.set_index('Unnamed: 0',drop=True)
    df_ = df_[~df_.index.duplicated(keep='first')]
    df_.dropna(axis=1, how='all',inplace=True)
    for column in list(df_.columns):
        if "Motion" in column:
            df_[column] = df_[column].apply(Fix_01)
        elif "Temperature" in column:
            df_[column] = df_.loc[:,column].apply(F_to_C) 

        if '[oF]' in column:
            df_.rename(columns={column:column.split()[0]},inplace=True)
        elif '[sec]':
            df_.rename(columns={column:column.split()[0]},inplace=True)

    df_[['T_ctrl','T_out','T_stp_cool','T_stp_heat']] = df_.loc[:,['T_ctrl','T_out','T_stp_cool','T_stp_heat']].apply(F_to_C)
    df_.index.name = ''
    df_.sort_index(inplace=True)
    return df_

def F_to_C(val):
    """
    Convert Farenheit value to Celsius
    
    Inputs:
    -------
    val -> numeric value (hopefully in F)

    Outputs:
    --------
    -> converted value
    """
    return (val-32.)*5./9.


def Fix_01(val):
    """
    Convert a 0.1 value to a value of 1. It was a mistake in this generation of DYD data. 
    
    Inputs:
    -------
    val -> float

    Outputs:
    --------
    -> converted value
    """    
    if val == 0.1:
        return 1
    else:
        return val

def Populate_Hour_and_Weekday(df_blank, freq='hour'):
    """
    Pull infromation from the datetimeindex 
    
    Inputs:
    -------
    df_blank -> a dataframe with a datetime index

    Outputs:
    --------
    -> dataframe with new columns for hour and weekdays
    """    
    df_new = df_blank.copy()
    if freq == 'hour':
        df_new['H_t'] = df_new.index.hour
    elif freq == '30M':
        df_new['H_t'] = list(map(add,list(df_new.index.hour), list(map(Bin_minutes, df_new.index.minute))))#
    df_new['W_t'] = df_new.index.weekday
    df_new['W_t'] = df_new.loc[:,'W_t'].apply(Is_This_A_Weekday)

    return df_new

def Bin_minutes(val):
    """
    Map minutes to a time bin. 
    """
    if val/60. >= 0.5:
        return 0.5
    else:
        return 0.0

def Is_This_A_Weekday(val):
    """
    Map day of week to binary 1 if weekday 0 if a weekend

    Inputs:
    -------
    val -> representing day of week

    Outputs:
    --------
    -> binary value for being a weekday
    """

    if val in [5,6]:
        return 0
    else:
        return 1


def Reduce_dataframe_to_key_columns(df_):
    """
    Format the interval data to be fed into the training

    Inputs:
    -------
    df_ -> dataframe

    Outputs:
    --------
    -> smaller dataframe
    """
    df_train_ = df_.filter(items=constant.COLUMN_VARIABLES)
    df_train_.at[:,'M_t'] = df_train_.loc[:,'M_t'].astype('int')
    return df_train_

def HomeOccupancyState(df_):
    """ Determine the occupancy state of the home based on all available sensors
    
    Inputs:
    -------
    df_ -> dataframe 

    Outputs:
    --------
    -> dataframe with a motion state column
    """
    occ_columns = [x for x in df_.columns.tolist() if "Motion" in x]
    df_['M_t'] = df_[occ_columns].any(axis=1,skipna=True)


def MapToSingleIncrease(val):
    """
    Need 30 minute values to be sequential for some of the tools(i.e. 1,2,3,4) so using a format
    like 5,10,15,20 won't work.

    Inputs:
    -------
    val -> minutes

    Outputs:
    --------
    -> mapped val
    """
    return val/5

def files(path, suffix='.xz'):
    """
    Generator to go through a directory and find all the files that have the correct file ending

    Inputs:
    -------
    path -> path

    Outputs:
    --------
    -> single file
    """
    for file in os.listdir(path):
        if suffix in file:
            if os.path.isfile(os.path.join(path, file)):
                yield file

def retrieve_tstat_id(file):
    """
    Remove thermostat id from file name

    Inputs:
    -------
    file -> file name

    Outputs:
    --------
    -> thermostat identifier string
    """
    tstat_id = file.split(".")[0]
    return tstat_id


def make_train_and_test_dataframes(df_, split_day, time_step_buffer=12):
    """
    Given a dataframe with a datetime index and a date to split on gnerate a train and test split
    Will look for 8 weeks prior and 2 weeks forward.

    Inputs:
    -------
    df_ -> dataframe
    split_day -> string of YYYY-MM-DD

    Outputs:
    --------
    -> train and test dataframe
    """

    split_day_datetime = pd.to_datetime(split_day)
    datetime_buffer = pd.Timedelta('30m')*time_step_buffer

    train_ = df_[split_day_datetime-pd.Timedelta('56D')-datetime_buffer:split_day_datetime-pd.Timedelta('30m')]
    test_ = df_[split_day_datetime-datetime_buffer:split_day_datetime+pd.Timedelta('14D')-pd.Timedelta('30m')]

    return train_, test_

def import_train_days():
    """
    Read in constant values defiend for train and test split

    Inputs:
    -------
    None

    Outputs:
    --------
    -> a list of values
    """
    return constant.TRAIN_TEST_SPLIT_VALS

def map_error_fnc(real_, predict_):
    """
    Handle comparisons involving nans for analaysis 

    Inputs:
    -------
    real_ -> measured value
    predict_ -> predicited value

    Outputs:
    --------
    -> a binary value of 1 if there is an error, 0 if not or Nan if being compared to a missing value.
    """
    if real_ == np.nan:
        return np.nan
    elif real_ == predict_:
        return 0
    else:
        return 1
