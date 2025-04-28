import os.path as osp
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import *

class NGSIM(Dataset):
    def __init__(self, cfgs, split):
        self.root = cfgs['root']
        self.eigenmaps_k = cfgs.get('eigenmaps_k', 8)
        self.similarity_delta = cfgs.get('similarity_delta', 0.1)

        # Load split data (train.pkl / val.pkl / test.pkl)
        with open(osp.join(self.root, f"{split}.pkl"), "rb") as f:
            self.data = pickle.load(f)

        self.num_nodes = self.data["x"].shape[2]
        self.num_features = self.data["x"].shape[3]
        self.in_len = self.data["x"].shape[1]
        self.out_len = self.data["y"].shape[1]

        if split == "train":
            self.complete_time_series = self.gen_complete_time_series()
            self.mean, self.std = self.compute_mean_std()

            graph_conn = self.gen_graph_conn()  # Just identity for now
            graph_sml = self.gen_graph_sml(self.complete_time_series)
            graph_cor = self.gen_graph_cor(self.complete_time_series)

            graphs = {
                "graph_conn": graph_conn,
                "graph_sml": graph_sml,
                "graph_cor": graph_cor
            }

            eigenmaps = self.gen_eigenmaps(graph_conn)
            transition_matrices = self.gen_transition_matrices(graphs)
            scaled_laplacian = compute_scaled_laplacian(graph_conn)

            # Convert everything to tensors
            tensors = [self.mean, self.std, self.complete_time_series, eigenmaps, transition_matrices, scaled_laplacian]
            self.mean, self.std, self.complete_time_series, eigenmaps, transition_matrices, scaled_laplacian = totensor(
                tensors, dtype=torch.float32
            )
            graphs = totensor(graphs, dtype=torch.float32)

            self.statics = {
                "eigenmaps": eigenmaps,
                "transition_matrices": transition_matrices,
                "graphs": graphs,
                "scaled_laplacian": scaled_laplacian
            }
    
    
    def __len__(self):
        return len(self.data['x'])


    def __getitem__(self, item):
        inputs = self.data['x'][item]
        targets = self.data['y'][item]

        inputs_time = self.time_transform(self.data['xtime'][item])
        targets_time = self.time_transform(self.data['ytime'][item])

        # Dummy rest flags (all 0s) to maintain compatibility
        inputs_rest = np.zeros_like(inputs_time)
        targets_rest = np.zeros_like(targets_time)

        inputs, targets = totensor([inputs, targets], dtype=torch.float32)
        inputs_time, targets_time, inputs_rest, targets_rest = totensor(
            [inputs_time, targets_time, inputs_rest, targets_rest], dtype=torch.int64
        )

        return inputs, targets, inputs_time, targets_time, inputs_rest, targets_rest


    def gen_complete_time_series(self):
        x, y = self.data['x'], self.data['y']
        full = np.concatenate((x, y), axis=1)  # [num_samples, in_len + out_len, N, C]
        complete_time_series = full.reshape(-1, self.num_nodes, self.num_features)  # [T, N, C]

        return complete_time_series
    
    
    def compute_mean_std(self):
        x = self.data['x']  # shape: [num_samples, in_len, N, C]
        x_flat = x.reshape(-1, self.num_features)
        mean = x_flat.mean(axis=0)
        std = x_flat.std(axis=0)

        return mean, std
    
    

