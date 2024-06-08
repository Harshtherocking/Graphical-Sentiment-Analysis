import spacy
from torch_geometric import edge_index
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

        # importing spacy model
        self.sm_eng_model = spacy.load("en_core_web_sm", enable=["tok2vec", "lemmatizer", "parser","morphologizer"])

        # get list of words and dependies
        self.update_paths()
        return None

    def update_paths (self): 
        self.dependencies = os.listdir(self.DepPath)
        self.words = os.listdir(self.WordPath)
        return None
    
    def transform (self, sentence : str) -> tuple[Data, tuple[list,list]]:
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
            if head.lemma_ != token.lemma_:
                # dependcy already present 
                if token.dep_ in self.dependencies: 
                    dep_tensor = torch.load(f = os.path.join(self.DepPath, token.dep_),weights_only= True).requires_grad = True
                # dependency not present
                else: 
                    dep_tensor = torch.randn(16, requires_grad=True)
                    torch.save(f= os.path.join(self.DepPath, token.dep_), obj = dep_tensor)
                    self.update_paths()

            # adding edge for tail to head 
            edge_index.append(tensor([token.i, head.i]))
            # adding edge attribute 
            edge_attr.append(dep_tensor)
            # saving dependency order
            edge_attr_order.append(token.dep_)
        
        # Data object creation
        x = torch.stack(x)
        edge_index = torch.stack(edge_index).t().contiguous()
        edge_attr = torch.stack(edge_attr)

        DependencyGraph = Data(x=x, edge_index= edge_index, edge_attr =edge_attr)

        return DependencyGraph, (x_order, edge_attr_order)




'''
----------------------------------------------------------------------------------------------------------------------
'''

# def preprocessing (sentence : str ): 
#     dependies = os.listdir(dep_path)
#     words = os.listdir(word_path)

#     doc = sm_eng_model(sentence.strip().lower())
#     # nodes tensor
#     x = []
#     for token in doc : 
#         if token.has_vector : 
#             x.append(tensor(token.vector, requires_grad = False ))
#         # if vector not in spacy model 
#         else :
#             if token.lemma_ in words : 
#                 word_tensor = torch.load(f= os.path.join(word_path, token.lemma_), weights_only= True)
#                 x.append(word_tensor)
#             else :
#                 temp_tensor = torch.rand((96,), requires_grad= True)
#                 x.append(temp_tensor)
#                 torch.save(f= os.path.join(word_path, token.lemma_), obj = temp_tensor)
#     # node tensor prepared 
#     x = torch.stack(x)
#     
#     # edge from tail to head
#     edge_index = []
#     # edge attribute tensor
#     edge_attr = []

#     for token in doc : 
#         tail = token 
#         head = token.head

#         if (head.lemma != token.lemma):
#             if token.dep_ in dependies :
#                 dep_tensor = torch.load(f = os.path.join(dep_path,token.dep_), weights_only= True)
#             else : 
#                 dep_tensor  = torch.rand(16, requires_grad=True)
#                 torch.save(f= os.path.join(dep_path, token.dep_), obj = dep_tensor)

#             # adding edge for tail to head 
#             edge_index.append(tensor([tail.i, head.i]))
#             # adding edge attribute 
#             edge_attr.append(dep_tensor)
#     # edge and attribute tensor prepared
#     edge_index = torch.stack(edge_index).t().contiguous()
#     edge_attr = torch.stack(edge_attr)
#     
#     # Homogenous Data object creation
#     graph = Data(x=x, edge_index = edge_index, edge_attr= edge_attr)
#             

'''
-----------------------------------------------------------------------------------------------------------------
'''

if __name__ == "__main__":
    preprocess = Preprocessor(dep_path = "dep-embed", word_path= "word-embed")
    sentence = "A huge dog bit me so hard that I almost got my bone cracked."
    graph , order = preprocess.transform(sentence)
    print(graph)
    print(order[0])
    print(order[1])
