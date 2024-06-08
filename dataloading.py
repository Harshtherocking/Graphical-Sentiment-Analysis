import torch 
from torch.utils.data import DataLoader, Dataset 
import pandas as pd
import os 

WORKDIR = os.getcwd()

class TextDataset (Dataset):
    def __init__ (self, dataset_path : str):
        assert dataset_path, "No Dataset Path provided"
        self.path = dataset_path
        self.data = pd.read_csv(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.loc[index][0]
        sentiment = self.data.loc[index][1]

        


if __name__ == "__main__": 
    data = os.path.join(WORKDIR, "Twitter_Data.csv")
    dataloader = TextDataset(data)
    print(dataloader.data.head())
    print(len(dataloader))
