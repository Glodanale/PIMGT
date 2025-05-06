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
        This approximates the deterministic part of dμ/dt from the LWR conservation law:
            ∂ρ/∂t + ∂f(ρ)/∂x = 0, where f(ρ) = ρ * u

        Args:
            mu: [B, N, F], where F includes [density (ρ), velocity (u)]
        Returns:
            d_mu/dt: shape [B, N, F], representing [dρ/dt, du/dt]
        """
        rho = mu[:, :, 0]  # [B, N] density
        u = mu[:, :, 1]    # [B, N] velocity
        f = rho * u        # [B, N] traffic flux f(p)

        # Compute spatial gradient of flux (∂f/∂x)
        df_dx = (f[:, 1:] - f[:, :-1]) / self.dx  # [B, N-1]
        df_dx = F.pad(df_dx, (0, 1), mode="replicate")  # Pad to maintain [B, N] shape

        drho_dt = -df_dx  # Conservation law: ∂ρ/∂t = -∂f/∂x
        du_dt = torch.zeros_like(u)  # Velocity evolution ignored in first-order LWR

        return torch.stack([drho_dt, du_dt], dim=-1)  # [B, N, F]


class KalmanRefinementLayer(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features

        # Linear system dynamics parameters
        self.A = nn.Parameter(torch.eye(num_nodes))  # System matrix (∂μ/∂t drift)
        self.B = nn.Parameter(torch.eye(num_nodes))  # Input/propagation matrix for process noise
        self.C = nn.Parameter(torch.ones(num_nodes))  # Diagonal noise scaling (used in Q)
        self.Q_scale = nn.Parameter(torch.tensor(0.01))  # Global noise scaling factor for Q

        # PDE dynamics
        self.use_pde_guidance = True  # can be toggled
        self.lwr = StochasticLWRLayer(dx=50.0)
        self.lam = nn.Parameter(torch.tensor(0.5))  # weight for PDE guidance (alpha)

    def forward(self, mu_t, W_t, dt):
        """
        Kalman prediction step with optional physics-informed adjustment (UDE-style).
        Implements a time-discretized version of:
            μ(t+1) ≈ μ(t) + dt * f(μ(t))
            W(t+1) ≈ W(t) + dt * dW/dt (Lyapunov-like equation)

        Args:
            mu_t: [B, N, F] - mean traffic state vector (μ)
            W_t: [B, N, N] - covariance matrix of densities (W)
            dt: scalar - timestep duration

        Returns:
            mu_next: [B, N, F] - updated mean
            W_next: [B, N, N] - updated covariance
        """
        # Auto-heal and validate shapes
        if mu_t.dim() != 3 or W_t.dim() != 3:
            raise ValueError(f"KalmanRefinementLayer: mu_t must be 3D (B, N, F) and W_t must be 3D (B, N, N), got mu_t {mu_t.shape}, W_t {W_t.shape}")

        B, N, F = mu_t.shape

        if W_t.shape[1] != N or W_t.shape[2] != N:
            raise ValueError(f"KalmanRefinementLayer: Mismatch between mu_t and W_t dimensions. mu_t has N={N}, W_t shape={W_t.shape}")

        if F < 2:
            raise ValueError(f"KalmanRefinementLayer: Expected at least 2 features (density and velocity), got {F} features.")

        # Handle common mistake: feature and node dimensions accidentally swapped
        if N < F and W_t.shape[1] == F:
            print("KalmanRefinementLayer: Auto-fixing mu_t shape [B, F, N] → [B, N, F]")
            mu_t = mu_t.permute(0, 2, 1)  # Swap axes 1 and 2

        # Main Kalman + PDE update
        A_batch = self.A.unsqueeze(0).expand(B, N, N)
        B_batch = self.B.unsqueeze(0).expand(B, N, N)
        C_diag = self.C.unsqueeze(0).expand(B, N)

        if self.use_pde_guidance:
            pde_adjustment = self.lwr(mu_t)  # [B, N, F]
        else:
            pde_adjustment = torch.zeros_like(mu_t)

        # State update (weighted combination of linear dynamics and PDE dynamics)
        # State update (with optional PDE correction)
        linear_drift = torch.bmm(A_batch, mu_t)  # (B, N, F)

        # Blend PDE and linear drift (UDE-style hybridization)
        if self.use_pde_guidance:
            pde_adjustment = self.lwr(mu_t)
            total_drift = (1 - self.lam) * linear_drift.clone()
            total_drift[:, :, :2] += self.lam * pde_adjustment
        else:
            total_drift = linear_drift

        # Euler step: μ(t+1) = μ(t) + dt * dμ/dt
        mu_next = mu_t + dt * total_drift

        # Covariance update  Process noise construction: Q = B C Bᵀ (C is diagonal)
        C_matrix = torch.diag_embed(C_diag)  # [B, N, N]
        process_noise = torch.bmm(torch.bmm(B_batch, C_matrix), B_batch.transpose(1, 2))  # [B, N, N]

        # Lyapunov-like covariance evolution:
        # dW/dt = A W + W Aᵀ + Q
        dW = torch.bmm(A_batch, W_t) + torch.bmm(W_t, A_batch.transpose(1, 2)) + self.Q_scale * process_noise
        
        # Euler step for covariance
        W_next = W_t + dt * dW  # [B, N, N]

        return mu_next, W_next

