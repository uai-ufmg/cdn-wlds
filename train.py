import os
import time

import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import open_data, open_multiple_data, get_folds, preprocess_data, Scaler

from configs.config import ConfigArgs

from train_files import train_wellgt, train_vae, train_byol, train_romanenkova


def set_seed(seed_number=42, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed_number)
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)


def train(cfg, train_data, val_data, device):

    match cfg['model_name'].lower():
        case "wellgt":
            train_wellgt(cfg, train_data, val_data, device)
        case "romanenkova":
            train_romanenkova(cfg, train_data, val_data, device)
        case "byol":
            train_byol(cfg, train_data, val_data, device)
        case "vae":
            train_vae(cfg, train_data, val_data, device)
        case _:
            raise NotImplementedError(f"Model not implemented! Model name {cfg['model_name'].lower()} is not an option.")


if __name__=='__main__':

    parser = ConfigArgs()

    cfg = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    set_seed(cfg['seed_number'])

    df = open_multiple_data(cfg['dataset_names'], cfg['data_dir'])

    if cfg['split_method'] == 'cv':
        splits = get_folds(cfg['dataset'])
        
        for i, (train_wells, val_wells) in enumerate(splits):
            train_data = df[df['WELL'].isin(train_wells)]
            val_data = df[df['WELL'].isin(val_wells)]

            train_data, val_data, scaler = preprocess_data(train_data, val_data, logs=cfg['columns_used'], q=[0.01, 0.99], verbose=cfg['verbose'])

            train(cfg, train_data, val_data, device)
            
    elif cfg['split_method'] == 'tt':
        train_wells, test_wells = train_test_split(data, test_size=0.15, shuffle=True, random_state=cfg['seed_number'])
        
        train_data = data[data['WELL'].isin(train_wells)]
        test_data = data[data['WELL'].isin(test_wells)]

        train_data, test_data, scaler = preprocess_data(train_data, test_data, logs=cfg['columns_used'], q=[0.01, 0.99], verbose=cfg['verbose'])

        train(cfg, train_data, val_data, device)
        
    elif cfg['split_method'] == 'tvt':
        raise NotImplementedError('Train-Validation-Test is not implemented for training')
        
    else:
        raise NotImplementedError('Data splitting method is not implemented for training')

