import spacy
from torch_geometric.data import Data
import torch
from torch import dtype, float32, tensor
# import numpy as np
# import json
import os 
#importing small scale model 
sm_eng_model = spacy.load("en_core_web_sm")

# path to find dependency vector 
dep_path = os.path.join(os.getcwd(), "dep-embed")
# path to find word vector
word_path = os.path.join(os.getcwd(),"word-embed")

# checking existence 
if not os.path.exists(dep_path): 
    os.mkdir(dep_path)
if not os.path.exists(word_path):
    os.mkdir(word_path)


def preprocessing (sentence : str ):
    
    dependies = os.listdir(dep_path)
    words = os.listdir(word_path)

    doc = sm_eng_model(sentence.strip().lower())
    # nodes tensor
    x = []
    for token in doc : 
        if token.has_vector : 
            x.append(tensor(token.vector, requires_grad = False ))
        # if vector not in spacy model 
        else :
            if token.lemma_ in words : 
                word_tensor = torch.load(f= os.path.join(word_path, token.lemma_), weights_only= True)
                x.append(word_tensor)
            else :
                temp_tensor = torch.rand((96,), requires_grad= True)
                x.append(temp_tensor)
                torch.save(f= os.path.join(word_path, token.lemma_), obj = temp_tensor)
    # node tensor prepared 
    x = torch.stack(x)
    
    # edge from tail to head
    edge_index = []
    # edge attribute tensor
    edge_attr = []

    for token in doc : 
        tail = token 
        head = token.head

        if (head.lemma != token.lemma):
            if token.dep_ in dependies :
                dep_tensor = torch.load(f = os.path.join(dep_path,token.dep_), weights_only= True)
            else : 
                dep_tensor  = torch.rand(16, requires_grad=True)
                torch.save(f= os.path.join(dep_path, token.dep_), obj = dep_tensor)

            # adding edge for tail to head 
            edge_index.append(tensor([tail.i, head.i]))
            # adding edge attribute 
            edge_attr.append(dep_tensor)
    # edge and attribute tensor prepared
    edge_index = torch.stack(edge_index).t().contiguous()
    edge_attr = torch.stack(edge_attr)
    
    # Homogenous Data object creation
    graph = Data(x=x, edge_index = edge_index, edge_attr= edge_attr)
    print(graph)

    torch.save(f= "graph", obj= graph )
    # with open ("graph", "wb") as file : 
    #     pickle.dump(graph,file)
            
if __name__ == "__main__":
    sentence = "Modi fkljfjsdffhj has a quick way to escape."
    preprocessing(sentence)

