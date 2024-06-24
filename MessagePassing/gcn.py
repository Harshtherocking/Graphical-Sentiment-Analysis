import torch
from torch_geometric.data import Data
from torch_geometric.utils  import degree, contains_isolated_nodes, add_remaining_self_loops 
from torch.nn import GRU, Linear, Parameter, Module, LSTM, AdaptiveMaxPool1d, AdaptiveAvgPool1d, Softmax, Tanh
from torch_geometric.nn import MessagePassing
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class myGCNConv (MessagePassing):
    def __init__ (self, in_channel: int, out_channel: int, attr_in_channel : int) : 
        super().__init__(aggr= "mean") 
        
        self.out_channel = out_channel

        self.A= Linear(in_channel, out_channel, bias = False)

        self.B= Linear(in_features= attr_in_channel, out_features= self.out_channel, bias = False)

        self.bias = Parameter(torch.empty(out_channel))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.bias.data.zero_()
        return super().reset_parameters()


    def forward(self, G : Data) : 
        x = G["x"]
        edge_attr =  G["edge_attr"]
        edge_index = G["edge_index"]
        
        # check for isolated nodes 
        # assert not contains_isolated_nodes(edge_index, x.shape[0]), "graph has isolated nodes" 
        assert edge_index.shape[1] == edge_attr.shape[0], f"Edges : {edge_index.shape[1]}, Attrs : {edge_attr.shape[0]}"

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
    def __init__(self, input_feature_size : int , output_feature_size : int, hid_size : int, dep_feature_size : int):
        super().__init__()

        self.hid_size = hid_size

        # graph convolution layer 
        self.gcn = myGCNConv(input_feature_size, output_feature_size,dep_feature_size)
        self.tanh= Tanh()

        # self.lstm = LSTM(input_size=output_feature_size, hidden_size=self.hid_size, bias = False, num_layers=3, batch_first= True)
        self.gru = GRU(input_size= output_feature_size, hidden_size= self.hid_size, bias = True, num_layers= 3, batch_first= True)

        # self.max_pool = AdaptiveMaxPool1d(1)
        # self.avg_pool = AdaptiveAvgPool1d(1)

        self.lin = Linear(out_features = 2, in_features= 3*self.hid_size + 2*hid_size)
        self.softmax = Softmax(dim=1)

        return None


    def forward (self, graphs: list[Data]): 
        batch_size = len(graphs)
        self.init_states(batch_size)
        seq = []
        lengths = []

        # graph convolution on each graph in the batch
        for g in graphs : 
            node = self.gcn(g)
            seq.append(self.tanh(node))
            lengths.append(node.shape[0])
        
        # padding of node list : x(i) has shape (length(i), out_features)
        padded_seq = pad_sequence(seq, batch_first = True)

        # Pack sequence object initialisation
        packed_seq = pack_padded_sequence(padded_seq,lengths = torch.tensor(lengths), batch_first= True, enforce_sorted= False)

        # passing pack sequence object 
        gru_out, h_t = self.gru(packed_seq, self.h0)

        # pooling the output
        gru_out, lengths = pad_packed_sequence(gru_out, batch_first = True)

        # adaptive average pooling by hand
        avg_pool = torch.sum(gru_out, dim=1)/lengths.view(-1,1)

        # adaptive max pooling by hand
        max_pool = torch.cat([torch.max(i[:l], dim=0)[0].view(1,-1) for i,l in zip(gru_out,lengths)], dim=0)
        
        # # concat pooling
        h_t_permuted = h_t.permute(1,0,2) 
        pool_concat = torch.concat([h_t_permuted.reshape(batch_size,-1),max_pool,avg_pool], dim=1)


        res = self.lin(pool_concat)
        return self.softmax(res)

    def init_states (self, batch_size):
        # self.c0 = torch.zeros(3,batch_size,self.hid_size)   
        self.h0 = torch.zeros(3,batch_size,self.hid_size)
        return None


# ------------------------------------------------------------------------

if __name__ == "__main__":
    
    graph = torch.load("graph", weights_only= False)

    batched_graph = [graph,graph]
    gnn = GcnDenseModel(input_feature_size =96, output_feature_size =32 ,hid_size =16, dep_feature_size =16)
    x = gnn(batched_graph)
    print(x)
