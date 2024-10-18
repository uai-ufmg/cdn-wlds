import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SinDataset(Dataset):
    def __init__(self, n_samples = 10000, seq_size = 256):
        """
            Arguments:
            ---------
                - n_samples (int): Number of samples in the dataset
                - seq_size (int): Sequences' size
            Return:
            ---------
                None
        """
        self.n_samples = n_samples
        self.seq_size = seq_size
        self.__create_dataset()

    def sin(self, x, a, b):
        """
        Generate the sinusoidal sequences
            Arguments:
            ---------
                - x (np.array): Array with values ranging from 0 to seq_size-1
                - a (int): Parameter used to give horizontal shift
                - b (int): Parameter used to determine the period
            Return:
            ---------
                - np.array: Array containing the sinusoidal sequence with the previously defined hyperparameters
        """
        e = np.random.normal(scale=0.05, size=len(x))
        #print(a,b,e)
        return np.sin(a + b*x) + e

    def __create_dataset(self):
        """
        Create the sinusoidal dataset
            Arguments:
            ---------
                None
            Return:
            ---------
                None
        """
        self.anchors = []
        self.positives = []
        self.negatives = []
        self.parameters = dict()
        self.parameters['a'] = []
        self.parameters['a2'] = []
        self.parameters['b'] = []
        self.parameters['name'] = []

        for i in range(self.n_samples):
            x = np.arange(self.seq_size)
            a = np.random.normal(scale=1000)
            b = abs(np.random.normal(loc=0.10, scale=0.05))
            first_sequence = self.sin(x, a, b)

            x = np.arange(self.seq_size)
            a2 = np.random.normal(scale=1000)
            b_positive = abs(b + np.random.normal(loc=0.005, scale=0.0002))
            second_sequence = self.sin(x, a2, b_positive)

            self.anchors.append(first_sequence)
            self.positives.append(second_sequence)
            self.parameters['a'].append(a)
            self.parameters['a2'].append(a2)
            self.parameters['b'].append(b)
            self.parameters['name'].append(i)

            x = np.arange(self.seq_size)
            a3 = np.random.normal(scale=1000)
            b_neg = abs(np.random.normal(loc=0.10, scale=0.05))
            third_sequence = self.sin(x, a3, b_neg)
            self.negatives.append(third_sequence)

        self.max_b = max(self.parameters['b'])
        self.min_b = min(self.parameters['b'])

    def __len__(self):
        """
        __len__ function from torch Dataset
            Arguments:
            ---------
                None
            Return:
            ---------
                - int: Dataset's size
        """
        return len(self.anchors)

    def __getitem__(self, idx):
        """
        __getitem__ function from torch Dataset
            Arguments:
            ---------
                - idx (int): Index to select an item from dataset
            Return:
            ---------
                - first_sequence (torch.tensor): Anchor sinusoidal sequence
                - second_sequence (torch.tensor): Positive sinusoidal sequence
                - b (float): Period parameter used to generate both anchor and positive
                - name (int): Unique identifier that indicates the original index from the dataset
        """

        first_sequence, second_sequence = self.anchors[idx], self.positives[idx]
        a, a2, b, name = self.parameters['a'][idx], self.parameters['a2'][idx], self.parameters['b'][idx], self.parameters['name'][idx]

        first_sequence = np.array(first_sequence)
        second_sequence = np.array(second_sequence)

        first_sequence = torch.from_numpy(first_sequence).float()
        second_sequence = torch.from_numpy(second_sequence).float()

        return first_sequence, second_sequence, b, name