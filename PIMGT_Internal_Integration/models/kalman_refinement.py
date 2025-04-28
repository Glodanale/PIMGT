import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticLWRLayer(nn.Module):
    def __init__(self, dx=50.0):
        super().__init__()
        self.dx = dx

    def forward(self, mu):
        """
        Compute the PDE-based drift term from the LWR model.
        mu: [B, N, F], where F includes density (0) and velocity (1)
        Returns: d_mu/dt of shape [B, N, F]
        """
        rho = mu[:, :, 0]  # [B, N]
        u = mu[:, :, 1]    # [B, N]
        f = rho * u        # [B, N]

        df_dx = (f[:, 1:] - f[:, :-1]) / self.dx  # [B, N-1]
        df_dx = F.pad(df_dx, (0, 1), mode="replicate")  # [B, N]

        drho_dt = -df_dx
        du_dt = torch.zeros_like(u)

        return torch.stack([drho_dt, du_dt], dim=-1)  # [B, N, F]


class KalmanRefinementLayer(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features

        self.A = nn.Parameter(torch.eye(num_nodes))
        self.B = nn.Parameter(torch.eye(num_nodes))
        self.C = nn.Parameter(torch.ones(num_nodes))  # diagonal noise scale
        self.Q_scale = nn.Parameter(torch.tensor(0.01))

        self.use_pde_guidance = True  # can be toggled
        self.lwr = StochasticLWRLayer(dx=50.0)
        self.lam = nn.Parameter(torch.tensor(0.5))  # weight for PDE guidance

    def forward(self, mu_t, W_t, dt):
        """
        mu_t: [B, N, F] - predicted mean
        W_t: [B, N, N] - predicted covariance
        dt: float - time step
        Returns:
            mu_next: [B, N, F]
            W_next: [B, N, N]
        """
        B, N, F = mu_t.shape
        A_batch = self.A.unsqueeze(0).expand(B, N, N)
        B_batch = self.B.unsqueeze(0).expand(B, N, N)
        C_diag = self.C.unsqueeze(0).expand(B, N)

        if self.use_pde_guidance:
            pde_adjustment = self.lwr(mu_t)
        else:
            pde_adjustment = torch.zeros_like(mu_t)

        # State update (with optional PDE correction)
        linear_drift = torch.bmm(A_batch, mu_t.transpose(1, 2)).transpose(1, 2)
        total_drift = (1 - self.lam) * linear_drift + self.lam * pde_adjustment
        mu_next = mu_t + dt * total_drift

        # Covariance update
        C_matrix = torch.diag_embed(C_diag)
        process_noise = torch.bmm(torch.bmm(B_batch, C_matrix), B_batch.transpose(1, 2))  # [B, N, N]

        dW = torch.bmm(A_batch, W_t) + torch.bmm(W_t, A_batch.transpose(1, 2)) + self.Q_scale * process_noise
        W_next = W_t + dt * dW

        return mu_next, W_next
