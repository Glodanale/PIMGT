import pickle
import os.path as osp
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.utils import *

class I80(Dataset):
    def __init__(self, cfgs, split):
        self.root = cfgs['root']
        self.num_nodes = 70
        self.num_features = 3
        self.in_len = 30
        self.out_len = 10
        self.num_intervals = 600  # Adjust if needed later (not critical unless needed for complete_time_series math)
        self.start_time = '00:00'  # Assume timestamps start from midnight unless otherwise
        self.interval = 1  # 1 second between each timestep (based on your data)
        self.eigenmaps_k = cfgs.get('eigenmaps_k', 8)
        self.similarity_delta = cfgs.get('similarity_delta', 0.1)

        with open(osp.join(self.root, f'{split}.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        self.restday = pd.read_csv(osp.join(self.root, 'restday.csv'), parse_dates=['time'], index_col='time')

        if split == 'train':
            complete_time_series = self.gen_complete_time_series()
            mean, std = self.compute_mean_std()
            graph_conn = self.gen_graph_conn()
            graph_sml = self.gen_graph_sml(complete_time_series)
            graph_sml_dtw = self.gen_graph_sml_dtw()
            graph_cor = self.gen_graph_cor()
            graphs = {'graph_conn': graph_conn, 'graph_sml': graph_sml,
                      'graph_sml_dtw': graph_sml_dtw, 'graph_cor': graph_cor}
            eigenmaps = self.gen_eigenmaps(graph_conn)
            transition_matrices = self.gen_transition_matrices(graphs)
            scaled_laplacian = compute_scaled_laplacian(graph_conn)
            self.mean, self.std, self.complete_time_series, eigenmaps, transition_matrices, scaled_laplacian = totensor(
                [mean, std, complete_time_series, eigenmaps, transition_matrices, scaled_laplacian],
                dtype=torch.float32)
            graphs = totensor(graphs, dtype=torch.float32)
            self.statics = {'eigenmaps': eigenmaps,
                            'transition_matrices': transition_matrices,
                            'graphs': graphs,
                            'scaled_laplacian': scaled_laplacian}

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, item):
        inputs = self.data['x'][item]
        targets = self.data['y'][item]

        inputs_time = self.time_transform(self.data['xtime'][item])
        targets_time = self.time_transform(self.data['ytime'][item])

        inputs_rest = self.rest_transform(self.data['xtime'][item])
        targets_rest = self.rest_transform(self.data['ytime'][item])

        inputs, targets = totensor([inputs, targets], dtype=torch.float32)
        inputs_time, targets_time, inputs_rest, targets_rest = totensor(
            [inputs_time, targets_time, inputs_rest, targets_rest], dtype=torch.int64)

        return inputs, targets, inputs_time, targets_time, inputs_rest, targets_rest

    def gen_complete_time_series(self):
        x, y = self.data['x'], self.data['y']
        num_samples = x.shape[0]  # number of samples
        m = self.num_intervals - self.in_len - self.out_len + 1
        d = int(num_samples / m)  # number of days

        z = np.concatenate((x, y), axis=1)

        temp = [np.concatenate(
            (z[(u * m):((u + 1) * m):(self.in_len + self.out_len)].reshape(-1, self.num_nodes, self.num_features),
             z[((u + 1) * m - m % (self.in_len + self.out_len) + 1):((u + 1) * m), -1]), axis=0)
                for u in range(d)]
        complete_time_series = np.concatenate(temp, axis=0)

        return complete_time_series

    def compute_mean_std(self):
        mean = self.data['x'].mean()
        std = self.data['x'].std()

        return mean, std

    def gen_graph_conn(self):
        with open(osp.join(self.root, 'graph_ngsimi-80_conn.pkl'), 'rb') as f:
            graph_conn = pickle.load(f).astype(np.float32)

        return graph_conn

    def gen_graph_sml(self, complete_time_series):
        x = complete_time_series.transpose((1, 0, 2)).reshape(self.num_nodes, -1)
        graph_sml = compute_graph_sml(x, delta=self.similarity_delta)

        return graph_sml

    def gen_graph_sml_dtw(self):
        with open(osp.join(self.root, 'graph_ngsimi-80_sml.pkl'), 'rb') as f:
            graph_sml_dtw = pickle.load(f).astype(np.float32)

        return graph_sml_dtw

    def gen_graph_cor(self):
        with open(osp.join(self.root, 'graph_ngsimi-80_cor.pkl'), 'rb') as f:
            graph_cor = pickle.load(f).astype(np.float32)

        return graph_cor

    def gen_eigenmaps(self, graph_conn):
        eigenmaps = compute_eigenmaps(graph_conn, k=self.eigenmaps_k)

        return eigenmaps

    def gen_transition_matrices(self, graphs):
        S_conn = row_normalize(add_self_loop(graphs['graph_conn']))
        S_sml = row_normalize(add_self_loop(graphs['graph_sml']))
        S_cor = row_normalize(add_self_loop(graphs['graph_cor']))

        S = np.stack((S_conn, S_sml, S_cor), axis=0)

        return S

    def time_transform(self, time):
        # Convert integers to numpy datetime64[ms] if needed
        if np.issubdtype(time.dtype, np.integer):
            time = time.astype('datetime64[ms]')

        dt = [t.astype('datetime64[s]').astype(datetime) for t in time]
        hour, minute = [int(s) for s in self.start_time.split(':')]
        dt = [t.replace(hour=hour, minute=minute) for t in dt]
        dt = np.array([np.datetime64(t) for t in dt])

        # (time - start_of_day) in minutes
        time_in_minutes = ((time - dt) / np.timedelta64(1, 'm')).astype(np.int64)

        # Rescale from 0-1439 minutes → 0-599 intervals
        time_ind = (time_in_minutes * 600) // 1440  # integer division

        # Clamp just in case
        time_ind = np.clip(time_ind, 0, 599)

        return time_ind


    def rest_transform(self, time):
        if np.issubdtype(time.dtype, np.integer):
            time = time.astype('datetime64[ms]')

        dt = [t.astype('datetime64[s]').astype(datetime) for t in time]
        dates = [t.strftime('%Y-%m-%d') for t in dt]
        rest_ind = self.restday.loc[dates].to_numpy().flatten().astype(np.int64)

        # Clamp restday to [0,1] just to be paranoid
        rest_ind = np.clip(rest_ind, 0, 1)

        return rest_ind



if __name__ == '__main__':
    cfgs = yaml.safe_load(open('cfgs/I80_MGT.yaml'))['dataset']
    train_set = I80(cfgs, split='train')
    val_set = I80(cfgs, split='val')
    test_set = I80(cfgs, split='test')
    batch = train_set[0]
