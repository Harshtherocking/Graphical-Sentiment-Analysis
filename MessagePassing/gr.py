import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt 

graph = torch.load("graph", weights_only=False)
graph = to_networkx(graph)
print (graph)

nx.draw(graph, with_labels = True)
plt.show()
