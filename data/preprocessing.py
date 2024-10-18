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