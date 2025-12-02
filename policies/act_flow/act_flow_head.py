
"""
Flow Matching action head for ACT-like policies.

- Drop-in replacement for a CVAE head.
- Trains with Conditional Flow Matching (CFM) on action chunks.
- Inference integrates the learned velocity field with an ODE solver (Euler / Heun).

Inputs/Shapes
-------------
context: (B, C_len, d_model)  # encoder tokens (cross-attended)
x0:      (B, H, A)            # clean action chunk (supervision)
Returns:
  - loss() -> torch.Tensor
  - sample() -> (B, H, A)

Author: ChatGPT (ACT-Diff/Flow minimal head)
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------

def exists(x):
    return x is not None

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) in [0, 1]
        returns: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device) * -1)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -------------------------
# Velocity network (Transformer with cross-attention to context)
# -------------------------

class TransformerVectorField(nn.Module):
    """
    Predicts velocity u_theta(x_t, t, context) with shape (B, H, A).
    Uses a light Transformer decoder over the action sequence and cross-attends to encoded context tokens.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        timestep_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(action_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(horizon, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.t_embed = SinusoidalTimeEmbedding(timestep_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(timestep_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(self, x_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t:     (B, H, A)
        context: (B, C_len, d_model)
        t:       (B,) in [0,1]
        returns: (B, H, A) velocity
        """
        B, H, A = x_t.shape
        x = self.in_proj(x_t)                           # (B, H, d_model)
        x = x + self.pos_emb.unsqueeze(0)               # (B, H, d_model)

        te = self.t_proj(self.t_embed(t))               # (B, d_model)
        x = x + te.unsqueeze(1)                         # broadcast to (B, H, d_model)

        y = self.decoder(x, context)                    # (B, H, d_model)
        y = self.out_norm(y)
        v = self.out_proj(y)                            # (B, H, A)
        return v


# -------------------------
# Flow Matching Head
# -------------------------

class FlowMatchingActionHead(nn.Module):
    """
    Conditional Flow Matching (CFM) head for action chunks.

    Path: linear interpolation between noise z ~ N(0, s^2 I) and data x0
      x_t = (1 - alpha(t)) * z + alpha(t) * x0,  t ∈ [0,1]
      v*(x_t, t | x0, z) = d/dt x_t = alpha'(t) * (x0 - z)

    Loss: E_{x0, z, t} [ || u_theta(x_t, t, c) - v* ||^2 ]

    Sampling: ODE solve from t=0 to 1 with x(0)=z using Euler or Heun (RK2).
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        prior_std: float = 1.0,
        timestep_dim: int = 128,
        cond_drop_prob: float = 0.1,   # classifier-free guidance (drop context) during training
        alpha_schedule: str = "linear",# alpha(t) shape function
        solver: str = "heun",          # "euler" or "heun"
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.prior_std = prior_std
        self.cond_drop_prob = cond_drop_prob
        self.alpha_schedule = alpha_schedule
        self.solver = solver

        self.vnet = TransformerVectorField(
            action_dim=action_dim,
            horizon=horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            timestep_dim=timestep_dim,
        )

    # ---- path helpers ----
    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.alpha_schedule == "linear":
            return t
        elif self.alpha_schedule == "cosine":
            # smooth start/end
            return 0.5 * (1 - torch.cos(torch.pi * t))
        elif self.alpha_schedule == "quad":
            return t * t
        else:
            raise ValueError(f"Unknown alpha_schedule: {self.alpha_schedule}")

    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        if self.alpha_schedule == "linear":
            return torch.ones_like(t)
        elif self.alpha_schedule == "cosine":
            return 0.5 * torch.pi * torch.sin(torch.pi * t)
        elif self.alpha_schedule == "quad":
            return 2 * t
        else:
            raise ValueError(f"Unknown alpha_schedule: {self.alpha_schedule}")

    # ---- training ----
    def loss(
        self,
        x0: torch.Tensor,              # (B, H, A) 规范化后的动作块
        context: torch.Tensor,         # (B, C_len, D) 编码器 tokens
        projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,   # (B, H)  True=有效, False=padding
    ) -> torch.Tensor:
        B, H, A = x0.shape
        device = x0.device

        # 采样噪声与时间
        z = torch.randn_like(x0) * self.prior_std
        t = torch.rand(B, device=device)  # [0,1]

        alpha_t = self._alpha(t).view(B, 1, 1)          # (B,1,1)
        alpha_dot_t = self._alpha_dot(t).view(B, 1, 1)  # (B,1,1)

        # 线性路径与真速度场
        x_t = (1 - alpha_t) * z + alpha_t * x0          # (B,H,A)
        v_star = alpha_dot_t * (x0 - z)                 # (B,H,A)

        # classifier-free guidance: 训练时随机丢条件
        if self.training and self.cond_drop_prob > 0.0:
            keep = (torch.rand(B, device=device) > self.cond_drop_prob).float().view(B, 1, 1)
            context = keep * context + (1 - keep) * torch.zeros_like(context)

        # 预测速度场
        v_pred = self.vnet(x_t, context, t)             # (B,H,A)
        if projector is not None:
            v_pred = projector(v_pred)

        # ====== 掩码加权 MSE ======
        if mask is None:
            return F.mse_loss(v_pred, v_star)
        # mask: (B,H) -> (B,H,1)
        m = mask.unsqueeze(-1).to(v_pred.dtype)
        se = (v_pred - v_star) ** 2                      # (B,H,A)
        num = (se * m).sum()
        den = (m.sum() * A).clamp_min(1.0)
        return num / den

    # ---- inference ----
    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,     # (B, C_len, d_model)
        steps: int = 8,
        cfg_scale: float = 0.0,    # classifier-free guidance scale
        projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        B = context.shape[0]
        device = context.device
        h = 1.0 / steps
        t_vals = torch.linspace(0.0, 1.0, steps + 1, device=device)  # [0, 1]

        # initial noise
        x = torch.randn(B, self.horizon, self.action_dim, device=device) * self.prior_std
        traj = [x] if return_intermediates else None

        # optional classifier-free guidance: precompute context null
        if cfg_scale > 0.0:
            context_null = torch.zeros_like(context)

        for s in range(steps):
            t = t_vals[s].expand(B)           # (B,)
            tp = t_vals[s + 1].expand(B)

            if cfg_scale > 0.0:
                v_c = self.vnet(x, context, t)
                v_0 = self.vnet(x, context_null, t)
                v = (1 + cfg_scale) * v_c - cfg_scale * v_0
            else:
                v = self.vnet(x, context, t)

            if self.solver == "euler":
                x_next = x + h * v
            elif self.solver == "heun":
                # predictor
                x_pred = x + h * v
                if cfg_scale > 0.0:
                    v_c2 = self.vnet(x_pred, context, tp)
                    v_02 = self.vnet(x_pred, context_null, tp)
                    v2 = (1 + cfg_scale) * v_c2 - cfg_scale * v_02
                else:
                    v2 = self.vnet(x_pred, context, tp)
                x_next = x + 0.5 * h * (v + v2)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")

            if exists(projector):
                x_next = projector(x_next)

            x = x_next
            if return_intermediates:
                traj.append(x)

        return (x, traj) if return_intermediates else x


# -------------------------
# Wiring helper (optional)
# -------------------------

def build_flow_head_from_config(config) -> FlowMatchingActionHead:
    """
    Convenience: create the head from a (HuggingFace-style) config object.
    Expect attributes:
        - action_dim
        - horizon or chunk_size
        - dim_model
        - nhead
        - num_layers
    """
    horizon = getattr(config, "horizon", None) or getattr(config, "chunk_size", None)
    head = FlowMatchingActionHead(
        action_dim=config.action_dim,
        horizon=horizon,
        d_model=config.dim_model,
        nhead=getattr(config, "nhead", 8),
        num_layers=getattr(config, "num_layers", 6),
        prior_std=getattr(config, "prior_std", 1.0),
        cond_drop_prob=getattr(config, "cond_drop_prob", 0.1),
        alpha_schedule=getattr(config, "alpha_schedule", "linear"),
        solver=getattr(config, "solver", "heun"),
    )
    return head
