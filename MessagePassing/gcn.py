import torch
from torch_geometric.data import Data
from torch_geometric.utils  import degree, contains_isolated_nodes, add_remaining_self_loops 
from torch.nn import GRU, Linear, Parameter, Module, LSTM, AdaptiveMaxPool1d, AdaptiveAvgPool1d, Softmax, Tanh, BatchNorm1d, TransformerEncoderLayer
from torch_geometric.nn import MessagePassing
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class GCN (MessagePassing):
    def __init__ (self, in_channel: int, out_channel: int, num_attr: int) : 
        super().__init__(aggr= "mean") 
        
        self.W1= Linear(in_channel, out_channel, bias = False)

        # matrix for dependency embeddings
        self.W2= Linear(in_features= num_attr, out_features=out_channel, bias = False)

        self.W3= Linear(in_features=2*out_channel, out_features= out_channel, bias= True)

        self.bias = Parameter(torch.empty(out_channel))
        

    def forward(self, G : Data) : 
        x = G["x"]
        edge_attr =  G["edge_attr"]
        edge_index = G["edge_index"]
        
        # linear transformation
        x = self.W1(x)
        edge_attr = self.W2(edge_attr)

        # adding self-loops
        edge_index, edge_attr = add_remaining_self_loops(
                edge_index = edge_index, 
                edge_attr = edge_attr, 
                fill_value= torch.ones(edge_attr.shape[1]), 
                num_nodes= x.shape[0]
                )

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
        

    def message (self, x_j, norm, edge_attr): 
        msg = norm.view(-1,1) * self.W3(torch.concat([x_j,edge_attr],dim=1))
        return msg 



# ---------------------------------------------------------
class GcnDenseModel (Module):
    def __init__(self, input_feature_size : int, output_feature_size : int, hid_size : int, num_dep : int):
        super().__init__()

        self.hid_size = hid_size

        # graph convolution layer 
        self.gcn = GCN(input_feature_size, output_feature_size,num_dep)

        # batch normalization
        self.batchNorm1 = BatchNorm1d(output_feature_size)

        # transformer encoder
        self.transformer_encoder = TransformerEncoderLayer(
                d_model = output_feature_size,
                nhead = 4,
                batch_first = True,
                dim_feedforward =  4 * output_feature_size,
                bias = True
                )
        
        
        # rnn layer 
        # self.gru = GRU(input_size= output_feature_size, hidden_size= self.hid_size, bias = True, num_layers= 3, batch_first= True)

        # batch normalization 
        self.batchNorm2 = BatchNorm1d(output_feature_size)

        # dense layer 
        # self.lin = Linear(out_features = 2, in_features= 3*self.hid_size + 2*hid_size)
        self.lin = Linear(out_features = 2, in_features= 2*output_feature_size)
        
        # batch normalization
        self.batchNorm3 = BatchNorm1d(2)
        
        self.softmax = Softmax(dim=1)

        self.tanh= Tanh()


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
        
        lengths = torch.tensor(lengths)
        # padding of node list : x(i) has shape (length(i), out_features)
        padded_seq = pad_sequence(seq, batch_first = True)

        # (batch_size, lengths, output)
        # applying batch norm
        padded_seq = self.batchNorm1(padded_seq.permute(0,2,1))
        padded_seq = padded_seq.permute(0,2,1)

        # Pack sequence object initialisation
        # packed_seq = pack_padded_sequence(padded_seq,lengths = torch.tensor(lengths), batch_first= True, enforce_sorted= False)

        # passing pack sequence object 
        # gru_out, h_t = self.gru(packed_seq, self.h0)

        # passing pack sequence object  
        encoder_out = self.transformer_encoder(padded_seq)

        # pooling the output
        # out, lengths = pad_packed_sequence(encoder_out, batch_first = True)

        # (batch_size, lengths, output)
        # appyling batch norm
        encoder_out = self.batchNorm2(encoder_out.permute(0,2,1)) 
        encoder_out = encoder_out.permute(0,2,1)

        # adaptive average pooling by hand
        avg_pool = torch.sum(encoder_out, dim=1)/lengths.view(-1,1)

        # adaptive max pooling by hand
        max_pool = torch.cat([torch.max(i[:l], dim=0)[0].view(1,-1) for i,l in zip(encoder_out,lengths)], dim=0)
        
        # # concat pooling
        # h_t_permuted = h_t.permute(1,0,2) 
        # pool_concat = torch.concat([h_t_permuted.reshape(batch_size,-1),max_pool,avg_pool], dim=1)
        pool_concat = torch.concat([max_pool,avg_pool],dim=1)

        # dense layer propogation
        res = self.lin(pool_concat)
        res = self.batchNorm3(res)

        return self.softmax(res)


    def init_states (self, batch_size):
        self.h0 = torch.zeros(3,batch_size,self.hid_size)
        return None


