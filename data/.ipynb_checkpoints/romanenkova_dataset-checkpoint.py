import numpy as np
from tqdm import tqdm
import random

import torch
from torch.utils.data import Dataset

class RomanenkovaDataset(Dataset):

    def __init__(self, df, columns_used, seq_size=512, interval_size=200, well_name_column_name = 'WELL'):
        """
            Arguments:
            ---------
                - df (pd.DataFrame): Well log data
                - columns_used (list[str]): List of logs used. Ex: GR, NPHI, ...
                - seq_size (int): Size of sequence sent to the model
                - interval_size (int): Size of the interval used to extract consecutive sequences
                - well_name_column_name (str): Name of the column that indicates the well name in the data
            Return:
            ---------
                None
        """
        
        self.seq_size = seq_size
        self.interval_size = interval_size
        self.columns_used = columns_used
        self.well_name_column_name = well_name_column_name
        
        self.list_of_wells = list(df[well_name_column_name].unique())
        
        self.dict_of_sequences, self.list_of_anchors, self.list_of_positives = self.__create_dataset(df)

        self.__remove_wells_with_no_sequences()
    
    def __remove_wells_with_no_sequences(self):
        """
        Removing wells that have no sequence without missing data.
            Arguments:
            ---------
                None
            Return:
            ---------
                None
        """
        dict_sequences_cpy = self.dict_of_sequences.copy()

        for wellname, well_sequences in dict_sequences_cpy.items():
            if len(well_sequences) == 0:
                _ = self.dict_of_sequences.pop(wellname, None)
                
    
    def __create_dataset(self, df):
        """
        Creates a dataset of valid anchor and positive sequences based on the original dataframe.
            Arguments:
            ---------
                - df (pd.DataFrame): Well log data
            Return:
            ---------
                - dict_of_sequences (dict): Dictionary with valid sequences where keys are the name of the wells
                - list_of_anchors (list): List of possible anchors among all wells
                - list_of_positives (list): Respective positive sequences for each anchor. Indexes match with list_of_anchors.
        """
        
        dict_of_sequences = dict()
        list_of_anchors = list()
        list_of_positives = list()

        for i in range(len(self.list_of_wells)):
            
            wellname = self.list_of_wells[i]
            dict_of_sequences[wellname] = []
        
        for i in tqdm(range(len(self.list_of_wells))):
            
            wellname = self.list_of_wells[i]
            well_df = df[df[self.well_name_column_name] == wellname]

            idx_null = [j for j,x in enumerate(well_df[self.columns_used].values) if np.isnan(x).any()]

            j=0
            while j < well_df.shape[0]-(2*self.seq_size)-1:
                
                sequence = well_df.iloc[j:j+(2*self.seq_size)]
                
                idx_null = [k for k,x in enumerate(sequence[self.columns_used].values) if np.isnan(x).any()]
                
                if idx_null == []:
                    anchor = sequence.iloc[:self.seq_size]
                    positive = sequence.iloc[self.seq_size:]

                    dict_of_sequences[wellname].append([wellname, anchor, f'{wellname}{j}'])
                    dict_of_sequences[wellname].append([wellname, positive, f'{wellname}{j+self.seq_size}'])
                    
                    if random.random() < 0.5:
                        list_of_anchors.append([wellname, anchor, f'{wellname}{j}'])
                        list_of_positives.append([wellname, positive, f'{wellname}{j+self.seq_size}'])
                    else:
                        list_of_anchors.append([wellname, positive, f'{wellname}{j+self.seq_size}'])
                        list_of_positives.append([wellname, anchor, f'{wellname}{j}'])
                    
                    j = j + self.seq_size + self.interval_size
                else:
                    j = j + idx_null[-1] + 1
        
        return dict_of_sequences, list_of_anchors, list_of_positives

    def __get_negative(self, wellname):
        """
        Get negative sequences for each anchor. Has to be from different well.
            Arguments:
            ---------
                - wellname (str): Name of the anchor's well
            Return:
            ---------
                - neg_wellname (str): Name of the well where the negative sequence was extracted
                - neg_sequence (list): List of possible anchors among all wells
                - neg_sample_name (list): Name of the negative sequence (composed of the name of the well, the beginning depth and the end depth)
        """
        # Randomly select new well
        neg_wellname = random.choice(list(self.dict_of_sequences.keys()))
        # In case the new well selected is the same as the anchor's
        while neg_wellname == wellname:
            neg_wellname = random.choice(list(self.dict_of_sequences.keys()))

        # Get a sequence from the new well
        neg_samples = self.dict_of_sequences[neg_wellname]
        neg_wellname, neg_sequence, neg_sample_name = random.choice(neg_samples)

        return neg_wellname, neg_sequence, neg_sample_name
    
    def __len__(self):
        return len(self.list_of_anchors)
    
    def __getitem__(self, idx):
        """
            Arguments:
            ---------
                - idx (int): Index for selecting a sample from the dataset
            Return:
            ---------
                - wellname (str): Name of the well from which the sequence is taken
                - anchor_sample_name (str): Name of the anchor sequence (composed of the name of the well, the beginning depth and the end depth)
                - anchor (torch.Tensor): Anchor well log sequence
                - positive (torch.Tensor): Positive well log sequence
                - negative (torch.Tensor): Negative well log sequence
        """
        
        wellname, anchor, anchor_sample_name = self.list_of_anchors[idx]
        wellname, positive, positive_sample_name = self.list_of_positives[idx]
        
        anchor = anchor[self.columns_used].to_numpy()
        anchor = np.reshape(anchor, (-1, len(self.columns_used)))

        positive = positive[self.columns_used].to_numpy()
        positive = np.reshape(positive, (-1, len(self.columns_used)))

        neg_wellname, negative, neg_sample_name = self.__get_negative(wellname)
        negative = negative[self.columns_used].to_numpy()
        negative = np.reshape(negative, (-1, len(self.columns_used)))
        
        anchor = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()
        
        return wellname, anchor_sample_name, anchor, positive, negative
