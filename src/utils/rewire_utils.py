import copy
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
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


# define function that allow carbons to fully connect with all atoms and create new dataset
def carbon_connect(dataset, sparse_dataset, threshold=3, p=0):
    new_dataset = []
    counter_dataset = []
    idx = 0
    for data in dataset:
        data_copy = copy.copy(data)
        sparse_data = sparse_dataset[idx]
        # pick out all the virtual edges in the graph
        virtual_edges_index = torch.index_select(data.edge_index, 1, torch.tensor(
            [i for i in range(data.edge_index.shape[1]) if torch.sum(data.edge_attr[i]) == 0]))

        # pick out all carbon nodes
        carbon_nodes = []
        for node in range(data.x.shape[0]):
            if data.x[node, 1] == 1:
                carbon_nodes.append(node)

        # if the number of carbons in a molecule is too small, no rewiring is performed
        if len(carbon_nodes) <= 3:
            new_dataset.append(data)
            counter_dataset.append(data)

        else:

            # crate carbon rewired graph that keeps only carbon related edges, add to new dataset
            carbon_edge_index = torch.index_select(virtual_edges_index, 1, torch.tensor(
                [i for i in range(virtual_edges_index.shape[1]) if
                 virtual_edges_index[0, i] in carbon_nodes or virtual_edges_index[1, i] in carbon_nodes]))
            carbon_edge_index, mask = dropout_edge(carbon_edge_index, p)
            carbon_edge_attr = torch.zeros(carbon_edge_index.shape[1], 4)
            data_copy.edge_index = torch.cat((sparse_data.edge_index, carbon_edge_index), dim=1)
            data_copy.edge_attr = torch.cat((sparse_data.edge_attr, carbon_edge_attr), dim=0)
            new_dataset.append(data_copy)

            # create countering random connection graph dataset
            data_copy_ = copy.copy(data)
            extra_edge_num = carbon_edge_index.shape[1] // 2
            virtual_edges_list = [virtual_edges_index[:, i].unsqueeze(dim=1) for i in
                                  range(virtual_edges_index.shape[1])]
            sampled_edge_index = []
            for j in range(extra_edge_num):
                sample = random.choice(virtual_edges_list)

                # check if repeated edges are sampled
                sample_np = sample.numpy()
                while np.any([np.array_equal(sample_np, tensor.numpy()) for tensor in sampled_edge_index]):
                    sample = random.choice(virtual_edges_list)
                    sample_np = sample.numpy()

                # append both directions
                sampled_edge_index.append(sample)
                sampled_edge_index.append(torch.tensor([sample[1], sample[0]]).unsqueeze(dim=1))

            if len(sampled_edge_index) > 0:
                sampled_edge_index = torch.cat(sampled_edge_index, dim=1)
                data_copy_.edge_index = torch.cat((sparse_data.edge_index, sampled_edge_index), dim=1)
                data_copy_.edge_attr = data_copy.edge_attr
                counter_dataset.append(data_copy_)

            else:
                counter_dataset.append(sparse_data)

        idx += 1

    return new_dataset, counter_dataset


# define function that only allow carbons to fully connect and create new dataset
def carbon_only_connect(dataset, sparse_dataset, threshold=3, p=0):
    new_dataset = []
    counter_dataset = []
    idx = 0
    for data in dataset:
        data_copy = copy.copy(data)
        sparse_data = sparse_dataset[idx]
        # pick out all the virtual edges in the graph
        virtual_edges_index = torch.index_select(data.edge_index, 1, torch.tensor(
            [i for i in range(data.edge_index.shape[1]) if torch.sum(data.edge_attr[i]) == 0]))

        # pick out all carbon nodes
        carbon_nodes = []
        for node in range(data.x.shape[0]):
            if data.x[node, 1] == 1:
                carbon_nodes.append(node)

        # if the number of carbons in a molecule is too small, no rewiring is performed
        if len(carbon_nodes) <= 3:
            new_dataset.append(data)
            counter_dataset.append(data)

        else:

            # crate carbon rewired graph that keeps only carbon-carbon edges, add to new dataset
            carbon_edge_index = torch.index_select(virtual_edges_index, 1, torch.tensor(
                [i for i in range(virtual_edges_index.shape[1]) if
                 virtual_edges_index[0, i] in carbon_nodes and virtual_edges_index[1, i] in carbon_nodes]))
            carbon_edge_index, mask = dropout_edge(carbon_edge_index, p)
            carbon_edge_attr = torch.zeros(carbon_edge_index.shape[1], 4)
            data_copy.edge_index = torch.cat((sparse_data.edge_index, carbon_edge_index), dim=1)
            data_copy.edge_attr = torch.cat((sparse_data.edge_attr, carbon_edge_attr), dim=0)
            new_dataset.append(data_copy)

            # create countering random connection graph dataset
            data_copy_ = copy.copy(data)
            extra_edge_num = carbon_edge_index.shape[1] // 2
            virtual_edges_list = [virtual_edges_index[:, i].unsqueeze(dim=1) for i in
                                  range(virtual_edges_index.shape[1])]
            sampled_edge_index = []
            for j in range(extra_edge_num):
                sample = random.choice(virtual_edges_list)

                # check if repeated edges are sampled
                sample_np = sample.numpy()
                while np.any([np.array_equal(sample_np, tensor.numpy()) for tensor in sampled_edge_index]):
                    sample = random.choice(virtual_edges_list)
                    sample_np = sample.numpy()

                # append both directions
                sampled_edge_index.append(sample)
                sampled_edge_index.append(torch.tensor([sample[1], sample[0]]).unsqueeze(dim=1))

            if len(sampled_edge_index) > 0:
                sampled_edge_index = torch.cat(sampled_edge_index, dim=1)
                data_copy_.edge_index = torch.cat((sparse_data.edge_index, sampled_edge_index), dim=1)
                data_copy_.edge_attr = data_copy.edge_attr

                counter_dataset.append(data_copy_)

            else:
                counter_dataset.append(sparse_data)

        idx += 1

    return new_dataset, counter_dataset


def carbon_rewiring(dataset, sparse_dataset, type):
    train_dataset_sparse = sparse_dataset[:1000]
    val_dataset_sparse = sparse_dataset[1000:2000]
    test_dataset_sparse = sparse_dataset[2000:3000]

    train_dataset = dataset[:1000]
    val_dataset = dataset[1000:2000]
    test_dataset = dataset[2000:3000]

    if type == "c2a":
        train_dataset_carbon, train_dataset_counter = carbon_connect(train_dataset, train_dataset_sparse)
        val_dataset_carbon, val_dataset_counter = carbon_connect(val_dataset, val_dataset_sparse)
        test_dataset_carbon, test_dataset_counter = carbon_connect(test_dataset, test_dataset_sparse)
    else:
        train_dataset_carbon, train_dataset_counter_only = carbon_only_connect(train_dataset, train_dataset_sparse)
        val_dataset_carbon, val_dataset_counter_only = carbon_only_connect(val_dataset, val_dataset_sparse)
        test_dataset_carbon, test_dataset_counter_only = carbon_only_connect(test_dataset, test_dataset_sparse)

    train_loader_carbon = DataLoader(train_dataset_carbon, batch_size=32, shuffle=True)
    val_loader_carbon = DataLoader(val_dataset_carbon, batch_size=32, shuffle=False)
    test_loader_carbon = DataLoader(test_dataset_carbon, batch_size=32, shuffle=False)

    return  train_loader_carbon, val_loader_carbon,   test_loader_carbon
