import spacy
from torch_geometric.data import Data
import torch

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.vocab import GloVe

from .depTokenizer import DepTokenizer

class Preprocessor (): 
    def __init__(self):

        # loading pretrained word embeddings
        self.glove = GloVe(name = "twitter.27B", dim=50);

        # importing spacy model
        self.sm_eng_model = spacy.load(
                "en_core_web_sm", 
                enable=["lemmatizer", "parser","tagger","morphologizer","attribute_ruler"]
                )
        
        # dependency tokenizer
        self.dep_tokenizer = DepTokenizer()
    

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
            dep = self.dep_tokenizer(token.dep_.lower())
            if dep is not None: 
                edge_attr.append(dep)
            else : 
                print(f"dependency one hot encoding not found for {token.dep_}")
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



if __name__ == "__main__":
    prep = Preprocessor()
    sentence = "I saw a dog toppled over the bench just like a bottle."
    data = prep(sentence)
    __import__('pprint').pprint(data)
