import os

import lasio
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from .data import Data


class Taranaki(Data):
    
    def __init__(self, directory:str, verbose:bool=False) -> None:
        """
            Arguments:
            ---------
                - directory (str): Path to the directory where data is
                - verbose (bool): If True, print progress details. Else, does not print anything.
        """
        
        self.directory = directory
        
        self.std_names = {'WELLNAME':'WELL',
                          'X':'X_LOC',
                          'Y':'Y_LOC',
                          'NEUT':'NPHI',
                          'DENS':'RHOB',
                          'RESD':'RDEP',
                          'RESM':'RMED',
                          'RESS':'RSHA'}
        
        self.data = self.open_data(verbose=verbose)
    
    def standardize_names(self, df:pd.DataFrame) -> pd.DataFrame:
        '''
        Change column names in the dataset to match the standard names.
            Arguments:
            ---------
                - df (pd.DataFrame): Well log dataset
            Return:
            ---------
                - df (pd.DataFrame): Well log dataset with standardized log names
        '''
        
        df = df.rename(columns=self.std_names)
        return df
    
    def open_data(self, verbose:bool) -> tuple[pd.DataFrame, LabelEncoder]:
        
        """
        Main method to open data.
            Arguments:
            ---------
                - verbose (bool): If True, print progress details. Else, does not print anything.
            Return:
            ---------
                - data (pd.DataFrame): Well log dataset fully configured to be used
                - le (LabelEncoder): Label Encoder used to encode lithology classes to consecutive numbers
        """
        
        df_wells_taranaki = pd.read_csv(self.directory)
        df_wells_taranaki = self.standardize_names(df_wells_taranaki)

        return df_wells_taranaki
