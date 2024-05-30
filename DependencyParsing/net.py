import networkx as nx
from torch_geometric.utils import to_networkx
import torch
import matplotlib.pyplot as plt

data_obj = torch.load(f = "graph", weights_only= False)

__import__('pprint').pprint(data_obj)

graph = to_networkx(data = data_obj)

nx.draw(graph)

plt.show()
