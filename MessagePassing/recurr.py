import torch
from torch.nn import LSTM
in_size = 4 # each word has 4 dimensions 
hid_size = 3
num_layers = 2
lstmlayer = LSTM(input_size=in_size, hidden_size=hid_size, num_layers=num_layers, bias = False)

h_o = torch.randn(num_layers, hid_size)
c_o = torch.randn(num_layers, hid_size)

# run 
input_sequence_1= torch.randn(10,in_size)
output, states= lstmlayer(input_sequence_1, (h_o,c_o))
h_t, c_t = states

print("for first sentence of length : ", input_sequence_1.shape[0])
print("output: ", output.shape)
print("h_t: ", h_t.shape)
print("c_t: ", c_t.shape)

#run
input_sequence_2= torch.randn(12,in_size)
output, states= lstmlayer(input_sequence_2, (h_o,c_o))
h_t, c_t = states

print("for second sentence of length : ", input_sequence_2.shape[0])
print("output: ", output.shape)
print("h_t: ", h_t.shape)
print("c_t: ", c_t.shape)
