import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SinDataset(Dataset):
    def __init__(self, n_samples = 10000, seq_size = 256):
        self.n_samples = n_samples
        self.seq_size = seq_size
        self.__create_dataset()

    def sin(self, x, a, b):
        e = np.random.normal(scale=0.05, size=len(x))
        #print(a,b,e)
        return np.sin(a + b*x) + e

    def __create_dataset(self):
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
        return len(self.anchors)

    def __getitem__(self, idx):

        first_sequence, second_sequence, third_sequence = self.anchors[idx], self.positives[idx], self.negatives[idx]
        a, a2, b, name = self.parameters['a'][idx], self.parameters['a2'][idx], self.parameters['b'][idx], self.parameters['name'][idx]

        first_sequence = np.array(first_sequence)
        second_sequence = np.array(second_sequence)
        third_sequence = np.array(third_sequence)
        
        #first_sequence = first_sequence.reshape(-1,1)
        #second_sequence = second_sequence.reshape(-1,1)
        #print(first_sequence.shape, second_sequence.shape)

        first_sequence = torch.from_numpy(first_sequence).float()
        second_sequence = torch.from_numpy(second_sequence).float()
        third_sequence = torch.from_numpy(third_sequence).float()

        return first_sequence, second_sequence, third_sequence, b, name
