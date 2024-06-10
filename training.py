import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os 

from DependencyParsing.graph_preprocess import Preprocessor

class TextDataset (Dataset):
    def __init__ (self, dataset_path : str, dep_path : str, word_path : str):
        assert dataset_path, "No Dataset Path provided"
        assert dep_path, "No DepEmbed Path provided"
        assert word_path, "No WordEmbed Path provided"
        self.path = dataset_path
        self.data = pd.read_csv(dataset_path)
        self.preprocess = Preprocessor(dep_path, word_path)
        self.len = len(self.data)
        return None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence = self.data.loc[index]["clean_text"]
        sentiment = torch.Tensor([self.data.loc[index]["category"]])
        try : 
            graph , order = self.preprocess(sentence)
            return graph,sentiment,order 
        except : 
            if index != self.len-1: 
                return self.__getitem__(index+1)
    
    def __iter__ (self):
        self.index = 0
        return self

    def __next__ (self):
        if self.index >= self.len : 
            raise StopIteration
        self.index +=1
        return self.__getitem__(self.index)

def collate(batch):
    graph = [item[0] for item in batch]
    sentiment = torch.stack([item[1] for item in batch])
    order = [item[2] for item in batch]
    return graph, sentiment, order


if __name__ == "__main__": 
    WORKDIR = os.getcwd()
    data = os.path.join(WORKDIR, "Twitter_Data.csv")
    depPath = os.path.join(WORKDIR, "Embeddings", "DepEmbed")
    wordPath = os.path.join(WORKDIR, "Embeddings", "WordEmbed")

    # Dataset object initialisation
    dataset = TextDataset(data, depPath, wordPath)

    # training and testing split 
    batch_size = 32
    train_data , test_data, val_data = random_split(dataset, [0.7,0.2,0,1])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True, collate_fn= collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = True, collate_fn= collate)
    val_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True, collate_fn= collate)

    for batch , out in enumerate(train_loader):
        print(batch)
        print(out[0][0])
        break
    
    # out = train_loader
    # print(out)
