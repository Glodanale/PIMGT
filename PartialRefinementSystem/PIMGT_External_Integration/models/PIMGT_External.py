import torch
import torch.nn as nn
from models.kalman_refinement import KalmanRefinementLayer
from models.MGT import MGT  # Original Meta Graph Transformer


class PIMGT_External(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.mgt = MGT(cfgs)
        self.kalman = KalmanRefinementLayer(
            num_nodes=cfgs['num_nodes'],
            num_features=cfgs['num_features']
        )
        self.dt = 0.1  # Time step (NGSIM = 0.1s)
        self.alpha = 0.5  # Blending weight between MGT and Kalman loss

    def forward(self, inputs, targets, *extras, **statics):
        mgt_outputs = self.mgt(inputs, targets, *extras, **statics)  # (B, Q, N, F)

        # Kalman roll-out
        B, P, N, F = inputs.shape
        W_t = torch.eye(N, device=inputs.device).unsqueeze(0).repeat(B, 1, 1) * 1e-3  # small initial uncertainty
        mu_t = inputs[:, -1, :, :]  # last timestep input

        kalman_outputs = []
        for i in range(mgt_outputs.shape[1]):  # out_len steps
            mu_t, W_t = self.kalman(mu_t, W_t, self.dt)
            kalman_outputs.append(mu_t)

        kalman_outputs = torch.stack(kalman_outputs, dim=1)  # (B, Q, N, F)

        return mgt_outputs, kalman_outputs
    
    
