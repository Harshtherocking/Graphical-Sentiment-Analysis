import spacy
import joblib
from torch_geometric.data import Data
import torch
import fasttext
from sklearn.preprocessing import OneHotEncoder

class Preprocessor (): 
    def __init__(self, ft_model_path:str, dep_encoder : OneHotEncoder):
        assert ft_model_path, "FastText Model not provided"
        assert dep_encoder, "Dependency Encoder not provided"

        self.dep_enc = dep_encoder

        # loading fast-text model
        self.ft_model = fasttext.load_model(ft_model_path)    

        # importing spacy model
        self.sm_eng_model = spacy.load(
                "en_core_web_sm", 
                enable=["lemmatizer", "parser","tagger","morphologizer","attribute_ruler"]
                )
    

    def __call__ (self, sentence : str) -> Data | None :
        # doc type conversion of sentence 
        doc = self.sm_eng_model(sentence.strip().lower())

        # Graph attributes  
        x = []
        edge_index = []
        edge_attr = []

        for token in doc :
            # word embedding from fasttext model 
            x.append(torch.tensor(self.ft_model[token.lemma_], dtype= torch.float))
            
            # dependency embedding from one hot encoder
            dep = self.dep_enc.transform([[token.dep_.lower()]]).toarray()
            if dep is not None: 
                edge_attr.append(torch.tensor(dep[0], dtype= torch.float))
            else : 
                print("dependency one hot encoding not found")
                return None
            
            # edge from token to head 
            head = token.head
            edge_index.append(torch.tensor([token.i, head.i], dtype= torch.long))


        # Data object creation
        x = torch.stack(x)
        edge_index = torch.stack(edge_index).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        # returning graph
        return  Data(x=x, edge_index= edge_index, edge_attr =edge_attr)

