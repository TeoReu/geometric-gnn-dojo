from torch_geometric.datasets import TUDataset, MoleculeNet

dataset = TUDataset(root='', name='PROTEINS', use_node_attr=True)

train = dataset[:100]
val = dataset[100:200]
test = dataset[200:300]