import copy
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch_geometric.utils import remove_self_loops, coalesce
from torch_sparse import SparseTensor

from src.utils.other import create_kchains
from src.utils.plot_utils import plot_2d, plot_3d

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def n_hop_plus_0_hop(tdata, n, p = 0):
    data = copy.copy(tdata)

    edge_index, edge_attr = data.edge_index, data.edge_attr
    N = data.num_nodes

    adj_0 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(N, N))
    adj_n = adj_0

    for i in range(n):
      adj_n = adj_n @ adj_0


    row, col, _ = adj_n.coo()


    edge_index2 = torch.stack([row, col], dim=0)
    edge_index2, _ = remove_self_loops(edge_index2)

    edge_index2, mask = dropout_edge(edge_index2, p)

    edge_index = torch.cat([edge_index, edge_index2], dim=1)
    if edge_attr is None:
        data.edge_index = coalesce(edge_index, num_nodes=N)
    else:
        # We treat newly added edge features as "zero-features":
        edge_attr2 = edge_attr.new_zeros(edge_index2.size(1),
                                          *edge_attr.size()[1:])
        edge_attr = torch.cat([edge_attr, edge_attr2], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N)
        data.edge_index, data.edge_attr = edge_index, edge_attr

    return data

def rewire_dataset_k0hop(dataset, k, p):
    new_dataset = []

    for data in dataset:
        new_dataset.append(n_hop_plus_0_hop(data, k, p))

    return new_dataset




