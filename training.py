import torch 
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
import pandas as pd
import os
import re
import joblib
from torch_geometric.data import Data 
from sklearn.preprocessing import OneHotEncoder 
from DependencyParsing.graph_preprocess import Preprocessor 
from MessagePassing.gcn import GcnDenseModel


# -------------------------------------------------------------------------------
class TextDataset (Dataset):
    def __init__ (self, dataset_path : str, ft_model_path : str, dep_enc_path : str, result_enc_path : str|None = None):
        assert dataset_path, "No Dataset Path provided"
        assert ft_model_path, "No FastText-Model Path provided"
        assert dep_enc_path, "No Dependecy Encoder Path provided"

        # data cleaning 
        self.data = pd.read_csv(dataset_path)
        self.data = self.data.dropna(axis=0,ignore_index= True)
        self.data.drop_duplicates(inplace= True, ignore_index=True)

        # regex 
        self.data["text"] = self.data["text"].apply(lambda x : self.clean_text(x))

        self.len = len(self.data)

        # one hot encoding for category
        if result_enc_path is None: 
            self.encoder = OneHotEncoder()
            self.encoder.fit(self.data["label"].unique().reshape(-1,1))
            # saving result encoder for testing
            with open(os.path.join("res-enc.bin"),"wb") as file:
                joblib.dump(self.encoder, file)
        else : 
            with open(result_enc_path, "rb") as file:
                self.encoder = joblib.load(file)

        # dependecy encoder 
        with open(dep_enc_path, "rb") as file :
            self.dep_encoder = joblib.load(file)

        # preprocessing objecton initialisation
        self.preprocess = Preprocessor(ft_model_path= ft_model_path, dep_encoder= self.dep_encoder)
        return None


    def clean_text (self, sent : str) -> str :
        sent = sent.lower() 
        patterns = [r'\((.*)\)', 
                    r'\<(.*)\>', 
                    r'[\\|\+|\=|\-|\_|\*|\/]',
                    r'[0-9]+'
                    ]
        for pattern in patterns :
            sent = re.sub(pattern,r"", sent) 
        return sent


    def __len__(self):
        return self.len


    def __getitem__(self, index) -> tuple[Data,torch.Tensor]:
        sentence = self.data.loc[index]["text"]
        sentiment = self.data.loc[index]["label"]

        #checking for valid sentence
        if not sentence :
            return self.__getitem__(index+1)
        sentiment = torch.tensor(self.encoder.transform([[sentiment]]).toarray()).reshape(-1)
        
        out = self.preprocess(sentence) 

        # checking for valid graph creation
        if out is not None: 
            return out , sentiment
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
    return graph,sentiment



# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    WORKDIR = os.getcwd()

    # data paths
    train_data = os.path.join(WORKDIR, "Train.csv")
    test_data = os.path.join(WORKDIR, "Test.csv")
    
      
    depPath = os.path.join(WORKDIR, "DependencyParsing","dep-encoder.bin")
    ft_modelPath = os.path.join(WORKDIR, "ft-model.bin")

    # Dataset object initialisation
    train_dataset = TextDataset(train_data,ft_modelPath,depPath)
    test_dataset = TextDataset(test_data,ft_modelPath,depPath, "res-enc.bin")


    batch_size = 32
    epochs = 15
    learning_rate = 1e-2

    
    # Module initialisation
    model = GcnDenseModel(
            input_feature_size= train_dataset[0][0]["x"].shape[1],
            output_feature_size= 16,
            hid_size= 8,
            num_dep = len(list(train_dataset.dep_encoder.get_feature_names_out()))
        )


    # loss function initialisation
    loss_fn = CrossEntropyLoss()

    # optimizer initialisation
    model_optim = Adam(params = model.parameters(), lr = learning_rate, amsgrad=True)

    
    # loading from model checkpoint if any
    try : 
        checkpoint = torch.load(os.path.join(WORKDIR, "checkpoints", "model1.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model_optim.load_state_dict(checkpoint["optimizer_state_dict"])
    except : 
        pass

    # data loader initialisation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, collate_fn= collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True, collate_fn= collate)

    
    # training function
    def train_loop(dataloader:DataLoader, model : torch.nn.Module, loss_fn, model_optim : Optimizer):
        # setting model to training mode
        model.train()
        size = len(dataloader.dataset)
        # iteration 
        for batch, out in enumerate(dataloader):
            X,y = out

            # model prediction 
            pred = model(X)
            
            # loss calculation
            loss = loss_fn(pred.type(torch.float),y)

            # backproporgation
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
            
            if batch % 100 == 0:
                loss, current = loss.item() , batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



    # testing function
    def test_loop (dataloader : DataLoader, model : torch.nn.Module, loss_fn):
        # setting model to evaluation mode
        model.eval()
        size = len(dataloader.dataset)

        num_batches = len(dataloader)
        test_loss , correct = 0,0 

        with torch.no_grad():
            for X,y in dataloader :
                # prediction 
                pred = model(X)
                # loss calculation
                test_loss +=loss_fn(pred,y).item()
                # correct prediction account
                correct += (pred.argmax()==y.argmax()).type(torch.float).sum().item()

        # average loss per batch 
        test_loss/= num_batches

        # for total acuracy
        correct /= size 

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


    for e in range(epochs):
        print(f"Epoch : {e+1} -----------------")
        train_loop(train_loader,model, loss_fn, model_optim)
        test_loop(test_loader,model,loss_fn)

        # saving checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optim.state_dict()
            }, os.path.join(WORKDIR, "checkpoints", f"model1.pt"))
        
        
    torch.save(f = os.path.join(WORKDIR, "models", f"model"), obj = model)
