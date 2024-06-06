import torch
from torch.nn import Module, LSTMCell, LSTM

class model (Module):
    def __init__ (self, input_size, hidden_size, num_layers):
        super().__init__(self)
        

# parameteric definations 
in_size = 4 # each word has 4 dimensions 
hid_size = 3
num_layers = 9

print("in_size: ", in_size)
print("hid_size: ", hid_size)
print("num_layers: ", num_layers)

lstmlayer = LSTM(input_size=in_size, hidden_size=hid_size, num_layers=num_layers, bias = False)

# input definations
# 10 words in a sentence, each word has 4 dimensional vector  
input_sequence = torch.randn(10,in_size)
print("input: ", input_sequence.shape)
h_o = torch.randn(num_layers, hid_size)
c_o = torch.randn(num_layers, hid_size)

# run 
output, states= lstmlayer(input_sequence, (h_o,c_o))
h_t, c_t = states

print("output: ", output.shape)
print("h_t: ", h_t.shape)
print("c_t: ", c_t.shape)
