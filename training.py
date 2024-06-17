import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer,SGD
import pandas as pd
import os 
from torch_geometric.data import Data 
from sklearn.preprocessing import OneHotEncoder 
from DependencyParsing.graph_preprocess import Preprocessor 
from MessagePassing.gcn import GcnDenseModel

# -------------------------------------------------------------------------------
class TextDataset (Dataset):
    def __init__ (self, dataset_path : str, dep_path : str, word_path : str):
        assert dataset_path, "No Dataset Path provided"
        assert dep_path, "No DepEmbed Path provided"
        assert word_path, "No WordEmbed Path provided"

        self.path = dataset_path
        
        self.data = pd.read_csv(dataset_path)
        self.data = self.data.dropna(axis=0,ignore_index= True)
        self.data.drop_duplicates(inplace= True, ignore_index=True)
        self.len = len(self.data)

        self.preprocess = Preprocessor(dep_path, word_path)

        # one hot encoding for category
        self.encoder = OneHotEncoder()
        self.encoder.fit(self.data["category"].unique().reshape(-1,1))
        return None

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> tuple[Data, torch.Tensor,tuple[list,list]]:
        sentence = self.data.loc[index]["clean_text"]
        sentiment = self.data.loc[index]["category"]
        if not sentence :
            return self.__getitem__(index+1)
        sentiment = torch.tensor(self.encoder.transform([[sentiment]]).toarray()).reshape(-1)
        
        out = self.preprocess(sentence) 
        if out is not None: 
            graph , order = out 
            return graph,sentiment,order 
        else : 
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



# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    WORKDIR = os.getcwd()
    data = os.path.join(WORKDIR, "Twitter_Data.csv")
    depPath = os.path.join(WORKDIR, "Embeddings", "DepEmbed")
    wordPath = os.path.join(WORKDIR, "Embeddings", "WordEmbed")

    # Dataset object initialisation
    dataset = TextDataset(data, depPath, wordPath)

    batch_size = 32
    epochs = 5
    learning_rate = 1e-3

    # Module initialisation
    model = GcnDenseModel(
            input_feature_size= dataset[0][0]["x"].shape[1],
            output_feature_size= 32,
            hid_size= 16,
            dep_feature_size= dataset[0][0]["edge_attr"].shape[1]
        )

    # loss function initialisation
    loss_fn = CrossEntropyLoss()
    # optimizer initialisation
    model_optim = SGD(params= model.parameters(), lr = learning_rate)

    # training and testing split 
    train_data , test_data = random_split(dataset, [0.8,0.2])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True, collate_fn= collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = True, collate_fn= collate)


    def update_and_save (embed : torch.Tensor, grad : torch.Tensor, lr : int|float, path : str) :
        new_embed = embed - lr*grad
        torch.save(f = path, obj = new_embed)
        return None

    
    def train_loop(dataloader:DataLoader, model : torch.nn.Module, loss_fn, model_optim : Optimizer):
        model.train()
        size = len(dataloader.dataset)
        for batch, out in enumerate(dataloader):
            X,y,order = out
            for graph in X : 
                for _,t in graph : 
                    if t.requires_grad : 
                        t.retain_grad()

            pred = model(X)
            loss = loss_fn(pred,y)

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
            
            if batch % 100 == 0:
                loss, current = loss.item() , batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


            # updating and saving embeddings
            for b_no, (word_order, dep_order) in enumerate(order):
                words= X[b_no]["x"]
                deps= X[b_no]["edge_attr"]
                word_grad = words.grad
                dep_grad = deps.grad

                # saving word embedding
                for idx,word in enumerate(word_order) :
                    if word : 
                        update_and_save(words[idx], word_grad[idx], path = os.path.join(wordPath, word), lr = 1e-2)
                
                # saving dep embedding
                for idx, dep in enumerate(dep_order):
                    if dep : 
                        update_and_save(deps[idx], dep_grad[idx], path = os.path.join(depPath, dep), lr = 1e-2)

    def test_loop (dataloader : DataLoader, model : torch.nn.Module, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss , correct = 0,0 

        with torch.no_grad():
            for out in dataloader :
                X,y,order = out
                pred = model(X)
                test_loss +=loss_fn(pred,y).item()
                correct += (pred.argmax()==y.argmax()).type(torch.float).sum().item()

        test_loss/= num_batches
        correct /= size 
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    for e in range(epochs):
        print(f"Epoch : {e+1} -----------------")
        train_loop(train_loader,model, loss_fn, model_optim)
        test_loop(test_loader,model,loss_fn)
        
    torch.save(f = os.path.join(WORKDIR, "models", f"model1"), obj = model)

