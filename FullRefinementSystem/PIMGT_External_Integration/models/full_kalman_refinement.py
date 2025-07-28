import torch
import torch.nn as nn
import torch.nn.functional as F


class FullStochasticLWRLayer(nn.Module):
    def __init__(self, dx=50.0):
        super().__init__()
        self.dx = dx

    def forward(self, mu):
        """
        Compute PDE-based drift terms including density, velocity, and cumulative flow.
        Args:
            mu: [B, N, F], where F includes [density (ρ), velocity (u), cumulative flow (q)]
        Returns:
            d_mu/dt: shape [B, N, F], representing [dρ/dt, du/dt, dq/dt]
        """
        rho = mu[:, :, 0]  # density
        u = mu[:, :, 1]    # velocity
        q = mu[:, :, 2]    # cumulative flow (treated for completeness)

        f = rho * u        # traffic flux

        # Compute spatial gradient of flux ∂f/∂x
        df_dx = (f[:, 1:] - f[:, :-1]) / self.dx
        df_dx = F.pad(df_dx, (0, 1), mode="replicate")

        drho_dt = -df_dx
        du_dt = torch.zeros_like(u)  # assume constant velocity evolution
        dq_dt = f  # cumulative flow increases with local flux

        return torch.stack([drho_dt, du_dt, dq_dt], dim=-1)


class FullKalmanRefinementLayer(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features

        self.A = nn.Parameter(torch.eye(num_nodes))
        self.B = nn.Parameter(torch.eye(num_nodes))
        self.C = nn.Parameter(torch.ones(num_nodes))
        self.Q_scale = nn.Parameter(torch.tensor(0.01))

        self.use_pde_guidance = True
        self.lwr = FullStochasticLWRLayer(dx=50.0)
        self.lam = nn.Parameter(torch.tensor(0.5))

    def forward(self, mu_t, W_t, dt):
        if mu_t.dim() != 3 or W_t.dim() != 3:
            raise ValueError("mu_t must be [B, N, F], W_t must be [B, N, N]")

        B, N, F = mu_t.shape
        if W_t.shape[1] != N or W_t.shape[2] != N:
            raise ValueError("Dimension mismatch between mu_t and W_t")

        if F < 3:
            raise ValueError("Expected at least 3 features: [ρ, u, q]")

        if N < F and W_t.shape[1] == F:
            print("Auto-fixing shape [B, F, N] → [B, N, F]")
            mu_t = mu_t.permute(0, 2, 1)

        A_batch = self.A.unsqueeze(0).expand(B, N, N)
        B_batch = self.B.unsqueeze(0).expand(B, N, N)
        C_diag = self.C.unsqueeze(0).expand(B, N)
        
        linear_drift = torch.bmm(A_batch, mu_t)

        if self.use_pde_guidance:
            pde_adjustment = self.lwr(mu_t)
            total_drift = (1 - self.lam) * linear_drift.clone()
            total_drift[:, :, :3] += self.lam * pde_adjustment
        else:
            pde_adjustment = torch.zeros_like(mu_t)
            total_drift = linear_drift
        
        mu_next = mu_t + dt * total_drift

        C_matrix = torch.diag_embed(C_diag)
        process_noise = torch.bmm(torch.bmm(B_batch, C_matrix), B_batch.transpose(1, 2))
        dW = torch.bmm(A_batch, W_t) + torch.bmm(W_t, A_batch.transpose(1, 2)) + self.Q_scale * process_noise
        W_next = W_t + dt * dW

        return mu_next, W_next
