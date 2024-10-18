import numpy as np
from tqdm import tqdm
import random

import torch
from torch.utils.data import Dataset

from .augmentation import Jitter, Scaling, MagWarp, TimeWarp

class WellLogDataset(Dataset):

    def __init__(self, df, columns_used, seq_size=512, interval_size=200, well_name_column_name='WELL', list_of_augmentations=[Jitter, Scaling, MagWarp, TimeWarp]):
        
        self.seq_size = seq_size
        self.interval_size = interval_size
        self.columns_used = columns_used
        self.well_name_column_name = well_name_column_name
        
        self.list_of_wells = list(df[well_name_column_name].unique())
        
        self.list_of_sequences = self.__create_dataset(df)
        
        self.list_of_augmentations = list_of_augmentations
        
    def __create_dataset(self, df):
        
        list_of_sequences = list()
        
        for i in tqdm(range(len(self.list_of_wells))):
            
            wellname = self.list_of_wells[i]
            well_df = df[df[self.well_name_column_name] == wellname]

            idx_null = [j for j,x in enumerate(well_df[self.columns_used].values) if np.isnan(x).any()]

            j=0
            while j < well_df.shape[0]-self.seq_size-1:
                
                sequence = well_df.iloc[j:j+self.seq_size]
                
                idx_null = [k for k,x in enumerate(sequence[self.columns_used].values) if np.isnan(x).any()]
                
                if idx_null == []:
                    list_of_sequences.append([wellname, sequence, f'{wellname}{j}'])
                    j = j + self.interval_size
                else:
                    j = j + idx_null[-1] + 1
        
        return list_of_sequences
    
    def apply_augmentation(self, sequence, columns_used):
        augmented_sample = sequence.copy()

        num_elements = random.randint(1, len(self.list_of_augmentations))
        random_augmentations = random.sample(self.list_of_augmentations, num_elements)

        for i, func in enumerate(random_augmentations):
            augmented_sample = func(augmented_sample, columns_used)

        return augmented_sample
            
    def __len__(self):
        return len(self.list_of_sequences)
    
    def __getitem__(self, idx):
        wellname, sequence, sample_name = self.list_of_sequences[idx]
        
        sequence_numpy = sequence[self.columns_used].to_numpy()
        sequence_numpy = np.reshape(sequence_numpy, (-1, len(self.columns_used)))
        positive_pair1 = self.apply_augmentation(sequence_numpy, columns_used=self.columns_used)
        positive_pair2 = self.apply_augmentation(sequence_numpy, columns_used=self.columns_used)
        
        well_data_torch = torch.from_numpy(sequence_numpy).float()
        positive_pair1_torch = torch.from_numpy(positive_pair1).float()
        positive_pair2_torch = torch.from_numpy(positive_pair2).float()
        
        return wellname, sample_name, well_data_torch, positive_pair1_torch, positive_pair2_torch