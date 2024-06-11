import torch
from torch_geometric.data import Data
from torch_geometric.utils  import degree, contains_isolated_nodes, add_remaining_self_loops 
from torch.nn import Linear, Parameter, Module, LSTM, AdaptiveMaxPool1d, AdaptiveAvgPool1d, Sigmoid, Softmax
from torch_geometric.nn import MessagePassing
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class myGCNConv (MessagePassing):
    def __init__ (self, in_channel: int, out_channel: int) : 
        super().__init__(aggr= "mean") 
        
        self.out_channel = out_channel

        # shape = (out_channel, in_channel)
        self.A= Linear(in_channel, out_channel, bias = False)

        self.B= Linear(in_features= edge_attr.shape[1], out_features= self.out_channel, bias = False)

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
                fill_value= torch.ones(edge_attr.shape[1]), num_nodes= x.shape[0])

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
        # linear transformation to (*, out_channel)
        # x_j_trans = torch.matmul(x_j, self.W_1.t())
        x_j_trans = self.A(x_j)
        # edge_attr_trans = torch.matmul(edge_attr, self.W_2.t())
        edge_attr_trans = self.B(edge_attr)
        # element wise multiplication and activation
        # multiplication with norm 
        msg = norm.view(-1,1) * torch.tanh(x_j_trans * edge_attr_trans)
        return msg 



# ---------------------------------------------------------
class GcnDenseModel (Module):
    def __init__(self, input_feature_size : int , output_feature_size : int, hid_size : int):
        super().__init__(self)

        self.hid_size = hid_size
        self.c0 = torch.randn(3,self.hid_size, requires_grad=True)
        self.h0 = torch.randn(3,self.hid_size, requires_grad=True)

        # graph convolution layer 
        self.gcn = myGCNConv(input_feature_size, output_feature_size)
        self.sigmoid = Sigmoid()

        self.lstm = LSTM(input_size=output_feature_size, hidden_size=self.hid_size, bias = False, num_layers=3, batch_first= True)

        self.max_pool = AdaptiveMaxPool1d(1)
        self.avg_pool = AdaptiveAvgPool1d(1)

        self.lin = Linear(out_features = 3, in_features= 3*self.hid_size)
        self.softmax = Softmax(dim=1)

        return None


    def forward (self, graph: list[Data]): 
        batch_size = len(graph)
        self.x = [g["x"] for g in graph]
        self.edge_attr = [g["edge_attr"] for g in graph]
        edge_index = [g["edge_index"] for g in graph]
        
        # graph convolution on each graph in the batch
        seq = []
        lengths = []
        for idx in range(batch_size):
            node = self.gcn(self.x[idx], self.edge_attr[idx], edge_index[idx])
            seq.append(self.sigmoid(node))
            lengths.append(node.shape[0])

        # padding of node list : x(i) has shape (length(i), out_features)
        padded_seq = pad_sequence(seq, batch_first = True)

        # Pack sequence object initialisation
        packed_seq = pack_padded_sequence(padded_seq,lengths = torch.tensor(lengths), batch_first= True, enforce_sorted= False)

        # passing pack sequence object 
        lstm_out, states = self.lstm(packed_seq, (self.h0,self.c0))
        h_t, c_t  = states

        # pooling the output
        lstm_out, lengths = pad_packed_sequence(lstm_out)
        max_pool = self.max_pool(lstm_out.permute(0,2,1)).view(batch_size,-1)
        avg_pool = self.avg_pool(lstm_out.permute(0,2,1)).view(batch_size,-1)

        # concat pooling
        pool_concat = torch.concat([h_t[-1],max_pool,avg_pool])
        
        # dense layer 
        res = self.lin(pool_concat)
        return self.softmax(res)
        

# ------------------------------------------------------------------------

if __name__ == "__main__":
    
    graph = torch.load("graph", weights_only= False)
    x = graph["x"]
    edge_attr = graph["edge_attr"]
    edge_index = graph["edge_index"]
