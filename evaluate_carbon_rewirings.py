import argparse

import pandas as pd
import os
import sys
import time
import random
import numpy as np

from scipy.stats import ortho_group

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, Sigmoid

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9
from torch_scatter import scatter

import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem import QED, Crippen, rdMolDescriptors, rdmolops
from rdkit.Chem.Draw import IPythonConsole

import py3Dmol
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.qm9_models import DimeNetPPModel, GVPGNNModel
from src.utils.other import CompleteGraph, SetTarget, seed
from src.utils.train_utils import run_experiment
from src.utils.rewire_utils import rewire_dataset_k0hop, carbon_rewiring


def main(args):

    seed(args.seed)

    RESULTS = {}
    DF_RESULTS = pd.DataFrame(columns=["Test MAE", "Val MAE", "Epoch", "Model", "Rewire", "P", "K", "seed"])

    if 'IS_GRADESCOPE_ENV' not in os.environ:
        path = './qm9'
        target = 0

        # Transforms which are applied during data loading:
        # (1) Fully connect the graphs, (2) Select the target/label
        transform = T.Compose([CompleteGraph(), SetTarget()])

        # Load the QM9 dataset with the transforms defined
        dataset = QM9(path, transform=transform)

        # Normalize targets per data sample to mean = 0 and std = 1.
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std

        # Load QM9 dataset with sparse graphs (by removing the full graphs transform)
        sparse_dataset = QM9(path, transform=SetTarget())

        # Normalize targets per data sample to mean = 0 and std = 1.
        mean = sparse_dataset.data.y.mean(dim=0, keepdim=True)
        std = sparse_dataset.data.y.std(dim=0, keepdim=True)
        sparse_dataset.data.y = (sparse_dataset.data.y - mean) / std

    if args.rewire == "c2a":
        train_loader, val_loader, test_loader = carbon_rewiring(dataset, sparse_dataset, "c2a")
    else:
        train_loader, val_loader, test_loader = carbon_rewiring(dataset, sparse_dataset, "c2c")


    if args.model == "gvp":
        model = GVPGNNModel(num_layers=2, emb_dim=64, in_dim=11, out_dim=1)
    elif args.model == "dime":
        model = DimeNetPPModel(num_layers=2, emb_dim=64, in_dim=11, out_dim=1)


    best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
        model,
        args.model,
        train_loader,
        val_loader,
        test_loader,
        args.rewire,
        'no_k',
        args.p,
        args.seed,
        n_epochs=100,
    )

    RESULTS[args.model] = (best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Test MAE", "Val MAE", "Epoch", "Model", "Rewire", "K", "P", "seed"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)

    DF_RESULTS = DF_RESULTS.loc[DF_RESULTS["Epoch"] % 10 == 0]

    DF_RESULTS.to_csv('results_carbon.cvs', mode='a', index=False, header=False)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--rewire', default="c2a",type=str)
    parser.add_argument('--p', type=float)
    parser.add_argument('--seed', type=int)


    args = parser.parse_args()
    main(args)




