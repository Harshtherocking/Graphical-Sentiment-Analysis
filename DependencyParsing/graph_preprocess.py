import spacy
from torch_geometric.data import Data
import torch
from sklearn.preprocessing import OneHotEncoder

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.vocab import GloVe

from . import dep_tokenizer 

class Preprocessor (): 
    def __init__(self, dep_encoder : OneHotEncoder):
        assert dep_encoder, "Dependency Encoder not provided"

        self.dep_enc = dep_encoder

        # loading pretrained word embeddings
        self.glove = GloVe(name = "twitter.27B", dim=50);

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
            with torch.no_grad():
                x.append(self.glove.get_vecs_by_tokens(token.lemma_))
            
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

