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
from src.utils.rewire_utils import rewire_dataset_k0hop


def main(args):

    seed(args.seed)

    RESULTS = {}
    DF_RESULTS = pd.DataFrame(columns=["Test MAE", "Val MAE", "Epoch", "Model", "Rewire", "P", "K", "seed"])

    if 'IS_GRADESCOPE_ENV' not in os.environ:
        path = './qm9'
        target = 0

        dataset = QM9(path, transform=SetTarget())

        # Normalize targets per data sample to mean = 0 and std = 1.
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        mean, std = mean[:, target].item(), std[:, target].item()


    if args.rewire == "k0hop":
        new_dataset = rewire_dataset_k0hop(dataset, args.k, args.p)
    else:
        new_dataset = dataset

    train_dataset = new_dataset[:11000]
    val_dataset = new_dataset[11000:12000]
    test_dataset = new_dataset[12000:13000]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
        args.k,
        args.p,
        args.seed,
        n_epochs=10,
    )

    RESULTS[args.model] = (best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Test MAE", "Val MAE", "Epoch", "Model", "Rewire", "K", "P", "seed"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)

    DF_RESULTS = DF_RESULTS.loc[DF_RESULTS["Epoch"] % 10 == 0]

    DF_RESULTS.to_csv('results.cvs', mode='a', index=False, header=False)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--rewire', type=str)
    parser.add_argument('--p', type=float)
    parser.add_argument('--k', type=int)
    parser.add_argument('--seed', type=int)


    args = parser.parse_args()
    main(args)




