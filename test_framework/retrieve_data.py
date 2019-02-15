
import pandas as pd
import numpy as np
import os

from . import constants as constant
from . import utils as util
import logging

logging.basicConfig(level=logging.INFO)

class thermostat_data():
    """
    Standard class for retrieving and processing interval data
    """
    def __init__(self,thermostat_identifier):
        """
        intialize the class
        """
        self.tstat_id = thermostat_identifier


    def retrieve_data_frame(self):
        """
        load the dataframe and apply certain cleaning.
        """
        self.df_raw = util.dataload_n_clean(self.tstat_id)


    def Form_data(self, markov_order=1):
        """
        Format data as generally required
        """
        var_to_keep = [x for x in constant.COLUMN_VARIABLES]

        self.df_raw = util.Populate_Hour_and_Weekday(self.df_raw,'30M')
        df_30min = util.Reduce_dataframe_to_key_columns(self.df_raw)

        df_30min = df_30min.groupby(pd.Grouper(freq='30min')).max()

        if markov_order > 1:
            for ord_ in range(1,markov_order):
                df_30min.at[:,'M_t-{}'.format(ord_)] = df_30min.loc[:,'M_t'].shift(+1*ord_)
                var_to_keep.append('M_t-{}'.format(ord_))

        df_30min = df_30min.dropna()
        df_30min.at[:,['H_t']] = df_30min.loc[:,['H_t']].apply(lambda x: util.MapToSingleIncrease(x)*10.)
        df_30min = df_30min[var_to_keep]
        df_30min = df_30min.dropna(axis=0)
        df_30min.at[:,:] = df_30min.astype('int')
        return df_30min



    def main(self):
        """
        main call which runs the formatting end to end
        """
        logging.info("thermostat {} - main execution for thermostat_data".format(self.tstat_id))

        self.retrieve_data_frame()
        logging.info("thermostat {} - retrieved csv data".format(self.tstat_id))

        util.HomeOccupancyState(self.df_raw)
        logging.info("thermostat {} - occupancy state added".format(self.tstat_id))

        self.df_clean = self.Form_data()
        logging.info("thermostat {} - standard formating complete".format(self.tstat_id))
