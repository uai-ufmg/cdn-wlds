# Import data_split, and delete_outliers functions for preprocessing data.
# Import Scaler class for scaling data.

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

from .data_force import Force
from .data_taranaki import Taranaki
from .data_petro import Petro
from .preprocessing import delete_outliers, get_folds
from .augmentation import Jitter, Scaling, MagWarp, TimeWarp
from .scaler import Scaler
from .well_dataset import WellLogDataset
from .sin_dataset import SinDataset


def open_data(dataset_name:str, data_dir:str, verbose:bool) -> pd.DataFrame:
    """
    Function that opens data according to the dataset wanted.
        Arguments:
        ---------
            - dataset_name (str): Name of the dataset (Force, Geolink, ...)
            - data_dir (str): Path for folder containing dataset
            - verbose (bool): If True, print progress details. Else, does not print anything.
        Return:
        ---------
            - data (pd.DataFrame): Well log dataset
    """
    
    dataset_name = (dataset_name).lower()
    
    if dataset_name == 'force':
        force_dataset = Force(data_dir, verbose)
        data = force_dataset.data
    elif dataset_name == 'taranaki':
        taranaki_dataset = Taranaki(data_dir, verbose)
        data = taranaki_dataset.data
    elif dataset_name == 'petro':
        petro_dataset = Petro(data_dir, verbose)
        data = petro_dataset.data
    else:
        raise NotImplementedError('Dataset name not supported')
        
    return data


def open_multiple_data(dataset_names:list[str], data_dirs:list[str], verbose:bool=False) -> pd.DataFrame:
    """
    Function that opens multiple data according to the datasets wanted.
        Arguments:
        ---------
            - dataset_names (list[str]): Name of the datasets (Force, Geolink, ...)
            - data_dirs (list[str]): List of paths to folders containing datasets
            - verbose (bool): If True, print progress details. Else, does not print anything.
        Return:
        ---------
            - data (pd.DataFrame): Well log dataset
    """
    assert len(dataset_names) == len(data_dirs), "Number of dataset names and directories must be the same"

    for i, dataset_name in enumerate(dataset_names):
        if i==0:
            df = open_data(dataset_name, data_dirs[i], verbose)
        else:
            data = open_data(dataset_name, data_dirs[i], verbose)
            df = pd.concat([df, data], ignore_index=True, sort=False)

    return df
    

def preprocess_data(data:pd.DataFrame, logs:list[str], random_state, q=[0.01, 0.99], scaler=None, verbose:bool=True):

    train_data = delete_outliers(df=train_data, logs=logs, q=q, verbose=verbose)
    test_data = delete_outliers(df=test_data, logs=logs, q=q, verbose=verbose)

    if scaler == None:
        scaler = Scaler()
        train_data[logs] = scaler.fit_transform(train_data[logs])
        test_data[logs] = scaler.transform(test_data[logs])
    else:
        train_data[logs] = scaler.transform(train_data[logs])
        test_data[logs] = scaler.transform(test_data[logs])

    return train_data, test_data, scaler

