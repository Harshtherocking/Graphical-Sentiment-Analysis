import torch
from torch_geometric.utils  import degree, contains_isolated_nodes, add_remaining_self_loops 
from torch.nn import Linear, Parameter, Module, LSTM
from torch_geometric.nn import MessagePassing


class myGCNConv (MessagePassing):
    def __init__ (self, in_channel: int, out_channel: int) : 
        super().__init__(aggr= "mean") 
        
        self.out_channel = out_channel

        # shape = (out_channel, in_channel)
        W_1 = Linear(in_channel, out_channel, bias = False).weight
        W_1.requires_grad = True
        self.W_1 = Parameter(W_1)

        W_2 = Linear(in_features= edge_attr.shape[1], out_features= self.out_channel, bias = False).weight
        W_2.requires_grad = True
        self.W_2 = Parameter(W_2)

        self.bias = Parameter(torch.empty(out_channel))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.bias.data.zero_()
        return super().reset_parameters()

    def forward(self, x , edge_index, edge_attr ) : 

        # check for isolated nodes 
        assert not contains_isolated_nodes(edge_index, x.shape[0]), "graph has isolated nodes" 

        # adding self-loops
        edge_index, edge_attr = add_remaining_self_loops(
                edge_index = edge_index, 
                edge_attr = edge_attr, 
                fill_value= torch.ones(edge_attr.shape[1]),
                num_nodes= x.shape[0])

        # computing normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) 
        deg_inv_sqrt = deg.pow(-0.5) # written in paper, so 1/2
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 

        # propogation 
        output = self.propagate(edge_index, x=x, norm=norm, edge_attr= edge_attr)
        output = output + self.bias
        
        return output
        
    def message(self, x_j, norm, edge_attr): 
        # linear tranformation to (*, out_channel)
        x_j_trans = torch.matmul(x_j, self.W_1.t())
        edge_attr_trans = torch.matmul(edge_attr, self.W_2.t())
        # element wise multiplication and activation
        # multiplication with norm 
        msg = norm.view(-1,1) * torch.tanh(x_j_trans * edge_attr_trans)
        return msg 

class GcnDenseModel (Module):
    def __init__(self, input_feature_size : int , output_feature_size : int, hid_size : int):
        super().__init__(self)
        # graph convolution layer 
        self.gcn = myGCNConv(input_feature_size, output_feature_size)

        # x will have shapes (length, output_feature_size)
        # x will need to change into shapes (max_length, output_feature_size)
        self.lstm = LSTM(input_size=output_feature_size, hidden_size=hid_size, bias = False, num_layers=3)

        # x has shape (max_length, hidden_size)
        return None




if __name__ == "__main__":
    
    graph = torch.load("graph", weights_only= False)
    
    x = graph["x"]
    edge_index = graph["edge_index"]
    edge_attr = graph["edge_attr"]

    print("before applying graph convolution : ", x.shape)
    print("edge attr : ", edge_attr.shape)

    gcn = myGCNConv(x.shape[1],32)
    x = gcn(x, edge_index, edge_attr)

    print("after applying graph convolution : ", x.shape)
    print("edge attr : ", edge_attr.shape)
