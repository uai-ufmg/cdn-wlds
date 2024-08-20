import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def delete_outliers(df:pd.DataFrame, logs:list[str], q=[0.01, 0.99], verbose:bool=True):
    """
    Function to apply winsorization (remove outliers by clipping extreme quartiles. Upper or lower quartiles).
    Arguments:
    ---------
        - original_data (pd.DataFrame): Well log data, including lithology column
        - logs (list[str]): List of log names used. Ex: GR, NPHI, ...
        - q (list[float]): List of percentiles to clip
        - verbose (bool): If True, print progress details. Else, does not print anything.
    Return:
    ---------
        - data (pd.DataFrame): Well log data without outliers.
    """
    
    num_cols = len(logs)
    
    for i, log in enumerate(logs):
        if verbose:
            print(f'Handling log {i + 1}/{num_cols} - {log}')
        if log == 'NPHI' and q[1]==0.99:
            if verbose:
                print(f'{log}: {df[log].quantile(0.02)}, {df[log].quantile(0.98)}')
            min_percentile = df[log].quantile(0.02)
            max_percentile = df[log].quantile(0.98)

            df.loc[df[log] < min_percentile, log] = np.nan
            df.loc[df[log] > max_percentile, log] = np.nan
        else:
            if verbose:
                print(f'{log}: {df[log].quantile(q[0])}, {df[log].quantile(q[1])}')
            min_percentile = df[log].quantile(q[0])
            max_percentile = df[log].quantile(q[1])

            df.loc[df[log] < min_percentile, log] = np.nan
            df.loc[df[log] > max_percentile, log] = np.nan

    if verbose:
        print()
    
    return df

def get_folds(dataset_name:str, fold_n:int=None):
    """
    Function to get list of wells in each fold.
    Arguments:
    ---------
        - dataset_name (str): Name of the dataset (used to generate the file containing the folds)
        - fold_n (int|None): Fold number to get. If None, select all folds (to cross-validate, for instance). If not None, select the fold number fold_n only (similar to a train-test environment).
    Return:
    ---------
        - splits (list): List of folds. Each element a fold: a list containing the names of wells in training and validation for that fold.
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
            raise ValueError(f'No folder with number {fold_n}')

    return splits