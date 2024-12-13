# Import data_split, and delete_outliers functions for preprocessing data.
# Import Scaler class for scaling data.

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

from .data_force import Force
from .data_taranaki import Taranaki
from .preprocessing import delete_outliers, preprocess_data
from .augmentation import Jitter, Scaling, MagWarp, TimeWarp
from .scaler import Scaler
from .well_dataset import WellLogDataset
from .sin_dataset import SinDataset
from .romanenkova_dataset import RomanenkovaDataset


def open_data(dataset_name:str, data_dir:str, verbose:bool) -> pd.DataFrame:
    
    """
    Function that opens data according to the dataset wanted.
        Arguments:
        ---------
            - dataset_name (str): Name of the dataset (Force, Taranaki, ...)
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
    else:
        raise NotImplementedError('Dataset name not supported')
        
    return data


def open_multiple_data(dataset_names:list[str], data_dirs:list[str], verbose:bool=False) -> pd.DataFrame:
    """
    Function that opens multiple data according to the datasets wanted.
        Arguments:
        ---------
            - dataset_names (list[str]): Name of the datasets (Force, Taranaki, ...)
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

def get_folds(dataset_name:str, fold_n:int=None):
    """
    Function that opens the fold files (used for cross-validation).
        Arguments:
        ---------
            - dataset_name (str): Name of the dataset (Force, Geolink, ...)
        Return:
        ---------
            - splits (list): List of splits. Either one fold or all folds.
    """
    
    if fold_n == None:
        with open(f'data/splits/{dataset_name}/splits.json', 'r') as f:
            splits = json.load(f)
    else:
        with open(f'data/splits/{dataset_name}/splits.json', 'r') as f:
            all_splits = json.load(f)
        try:
            splits = [all_splits[fold_n]]
        except:
            raise ValueError(f'No fold with number {fold_n}')

    return splits

def preprocess_data(train_data:pd.DataFrame, test_data:pd.DataFrame, logs:list[str], q=[0.01, 0.99], scaler=None, verbose:bool=True):
    """
    Function that preprocess already split data (winsorization and scaling).
        Arguments:
        ---------
            - train_data (pd.DataFrame): Well log train dataset
            - test_data (pd.DataFrame): Well log test dataset
            - logs (list[str]): List of logs used (GR, NPHI, ...)
            - q (list[float]): List of percentiles to clip in winsorization
            - scaler: Scaler object used (if it is already fitted)
            - verbose (bool): If True, print progress details. Else, does not print anything.
        Return:
        ---------
            - train_data (pd.DataFrame): Well log train dataset preprocessed
            - test_data (pd.DataFrame): Well log test dataset preprocessed
            - scaler: Fitted scaler
    """

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

