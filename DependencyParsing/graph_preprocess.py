import spacy
from torch_geometric.data import Data
import torch
from torch import tensor 
import os 

class Preprocessor (): 
    def __init__(self, dep_path : str, word_path : str):
        assert dep_path , "Dependency Path not provided"
        assert word_path, "Word Path not provided"
        
        self.DepPath = dep_path
        self.WordPath = word_path

        # checking path 
        if not os.path.exists(dep_path):
            os.mkdir(dep_path)
        if not os.path.exists(word_path):
            os.mkdir(word_path)

        self.update_paths()

        # importing spacy model
        self.sm_eng_model = spacy.load("en_core_web_sm", enable=["tok2vec", "lemmatizer", "parser","tagger","morphologizer","attribute_ruler"])

    # get list of words and dependies
    def update_paths (self): 
        self.dependencies = os.listdir(self.DepPath)
        self.words = os.listdir(self.WordPath)
        return None
    
    def __call__ (self, sentence : str) -> tuple[Data, tuple[list,list]] | None :
        # doc type conversion of sentence 
        doc = self.sm_eng_model(sentence.strip().lower())
        # Graph attributes  
        x = []
        edge_index = []
        edge_attr = []

        # ordered list for saving the tensor after training
        x_order = []
        edge_attr_order = []

        for token in doc :
            # word has vector
            if token.has_vector : 
                x.append(tensor(token.vector, requires_grad= False))
                x_order.append(False)
            # word doesn't have vector
            else : 
                # word already present
                if token.lemma_ in self.words : 
                    word_tensor = torch.load(f = os.path.join(self.WordPath, token.lemma_), weights_only= True).requires_grad = True
                    x.append(word_tensor)
                # word not present
                else :
                    word_tensor = torch.randn((96,), requires_grad= True)
                    x.append(word_tensor)
                    torch.save(f= os.path.join(self.WordPath, token.lemma_), obj= word_tensor)
                    self.update_paths()
                # saving word order
                x_order.append(token.lemma_)
            
            head = token.head
            # to avoid loop conflict 
            if head != token:
                # dependcy already present 
                if token.dep_ in self.dependencies: 
                    dep_tensor = torch.load(f = os.path.join(self.DepPath, token.dep_), weights_only= True)
                    dep_tensor.requires_grad = True
                    edge_attr.append(dep_tensor)
                # dependency not present
                else: 
                    dep_tensor = torch.randn((16,), requires_grad=True)
                    edge_attr.append(dep_tensor)
                    torch.save(f= os.path.join(self.DepPath, token.dep_), obj = dep_tensor)
                    self.update_paths()

                # saving order of dependencies
                edge_attr_order.append(token.dep_)

                # adding edge for tail to head 
                edge_index.append(tensor([token.i, head.i]))
        
        if len(edge_index) and len(x) and len(edge_attr): 
            pass
        else :
            return None
        # Data object creation
        x = torch.stack(x)
        edge_index = torch.stack(edge_index).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        if edge_index.shape[1] != edge_attr.shape[0]: 
            print(doc)
            print(f"{edge_index.shape[1]} ----- {edge_attr.shape[0]}")
            return None
    
        DependencyGraph = Data(x=x, edge_index= edge_index, edge_attr =edge_attr)

        return DependencyGraph, (x_order, edge_attr_order)


# --------------------
if __name__ == "__main__":
    preprocess = Preprocessor(dep_path = "dep-embed", word_path= "word-embed")
    sentence = "A huge dog bit me so hard that I almost got my bone cracked."
    out = preprocess(sentence)
    if out is not None : 
        graph, order = out
        print(graph)
        print(order[0])
        print(order[1])

