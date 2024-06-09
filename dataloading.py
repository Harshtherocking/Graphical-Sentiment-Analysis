import torch 
from torch.utils.data import Dataset 
import pandas as pd
import os 

from DependencyParsing.graph_preprocess import Preprocessor

WORKDIR = os.getcwd()

class TextDataset (Dataset):
    def __init__ (self, dataset_path : str, dep_path : str, word_path : str):
        assert dataset_path, "No Dataset Path provided"
        self.path = dataset_path
        self.data = pd.read_csv(dataset_path)
        self.preprocess = Preprocessor(dep_path, word_path)
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.loc[index]["clean_text"]
        sentiment = torch.Tensor([self.data.loc[index]["category"]])
        
        graph , order = self.preprocess(sentence)
        return graph,sentiment,order 


if __name__ == "__main__": 
    data = os.path.join(WORKDIR, "Twitter_Data.csv")

    depPath = os.path.join(WORKDIR, "DependencyParsing", "DepEmbed")
    wordPath = os.path.join(WORKDIR, "DependencyParsing", "wordEmbed")

    dataloader = TextDataset(data, depPath, wordPath)
    print(dataloader.data.head())
    print(len(dataloader))

    print(dataloader[9])
