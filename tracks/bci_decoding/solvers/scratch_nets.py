"""End-to-end EEG networks trained from scratch (random initialization).

``model`` picks the architecture; every network gets the same recipe as the
foundation-model ``scratch`` runs (REVE / LaBraM / CBraMod / BIOT / EEGPT /
BENDR): AdamW (lr 5e-4, weight decay 0.05), batch 16, 20 epochs, all
labeled training windows. Each window is z-scored per channel before the
network sees it.

Exceptions, each from the model's official code:

Each window is z-scored per channel except for RGNN and EEGMamba, which
take raw microvolts.

- ``MAtt`` (Pan et al., NeurIPS 2022; https://github.com/CECNL/MAtt, MIT
  license, Copyright (c) 2022 CECNL) is inlined below with its official
  BCI-IV-2a configuration (spatial conv sized to the task's channels). It
  keeps its official optimizer: Adam (weight decay 0.1) wrapped in the
  Stiefel-manifold step that keeps its attention projections orthonormal,
  and runs on CPU like the official code (its SVD layers loop per sample).

- ``RGNN`` (Zhong et al., IEEE TAFFC 2020; https://github.com/
  zhongpeixiang/RGNN) is its official ``SymSimGCNNet`` core rewritten in
  dense PyTorch (no torch_geometric): learnable symmetric adjacency with
  absolute-degree normalization, K-hop simple graph convolution, ReLU, sum
  pooling over channels, dropout, linear. Node features are differential
  entropy in 5 bands (delta 1-4, theta 4-8, alpha 8-14, beta 14-31, gamma
  31-50 Hz), standardized with training-set statistics. The adjacency starts
  from the paper's init, ``min(1, 5 / d^2)`` (d in cm, standard 10-05
  positions), with -1 added to left/right symmetric pairs. The paper's
  NodeDAT and EmotionDL regularizers are not used (NodeDAT needs unlabeled
  test-subject data, EmotionDL is for emotion label distributions; the repo
  omits both too). K=2, 64 hidden units and dropout 0.5 are our choices.

- ``EEGMamba`` (Wang et al., Neural Networks 2025; https://github.com/
  wjq-learning/EEGMamba, MIT, (c) 2025 Jiquan Wang) is its official
  architecture trained from random init: CBraMod-style patch embedding
  (1-s patches at 200 Hz, microvolts / 100), 12 bidirectional Mamba-2
  layers (d_model 200, headdim 50, d_state 64; the sequence is flipped
  after every layer), RMSNorm, and the official ``all_patch_reps`` head.
  ``mamba-ssm``'s CUDA kernels are replaced by a pure-PyTorch Mamba-2 (the
  chunked SSD scan from the Mamba-2 reference code), so it is slower but
  computes the same function.

- ``DualTSST`` (Li et al., arXiv 2409.03251) has no public code; it is
  reimplemented from the paper (Table I and Sec. III). Stated in the paper:
  Branch I on raw EEG (time conv 1x30, D1=40 -> BN -> depthwise spatial conv
  -> BN -> ELU -> avg-pool 1x120 stride 12 -> pointwise conv to D2=120);
  Branch II on the Morlet wavelet transform (n_cycles = freq / 2) in two
  views, [ch x F x T] and [F x ch x T], each time conv 1x125 -> separable
  conv over F or ch -> BN -> ELU -> avg-pool 1x64 stride 32 -> pointwise
  conv to 120; tokens concatenated with a learnable positional embedding;
  4-layer, 10-head Transformer; global average pooling; 2-layer MLP. Our
  choices where the paper is silent: resample to 250 Hz (its kernels are in
  samples at 250 Hz), wavelet frequencies 1-40 Hz in 1-Hz steps (amplitude),
  BN also after each time conv, feed-forward size 480, Transformer dropout
  0.1, MLP hidden size 64, no dropout in the CNN branches. The paper's
  Segment & Reassemble augmentation and its 1000-epoch schedule are not
  used.

Output folders and W&B runs are named ``<model>-scratch-shotall-seed<seed>``.
If W&B is installed and logged in, per-epoch training loss/accuracy and the
held-out benchmark metrics are logged to ``neurips-eeg-2026-bci`` (override
with ``WANDB_PROJECT``; ``WANDB_MODE=disabled`` turns it off).
"""

import math
import os

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

WANDB_PROJECT = "neurips-eeg-2026-bci"


# ---------------------------------------------------------------------------
# MAtt — ported from https://github.com/CECNL/MAtt (MIT, (c) 2022 CECNL).
# Changes: device-agnostic buffers, task-sized spatial conv / classifier,
# deprecated ``saved_variables`` -> ``saved_tensors``. Math is unchanged.
# ---------------------------------------------------------------------------

class StiefelParameter(nn.Parameter):
    """Parameter constrained to the Stiefel manifold (orthonormal columns)."""

    def __new__(cls, data=None, requires_grad=True):
        return super().__new__(cls, data, requires_grad=requires_grad)


def _symmetric(A):
    return 0.5 * (A + A.transpose(-1, -2))


def _orthogonal_projection(A, B):
    return A - B.mm(_symmetric(B.transpose(0, 1).mm(A)))


def _retraction(A, ref):
    Q, R = torch.linalg.qr(A + ref)
    sign = (R.diag().sign() + 0.5).sign().diag()
    return Q.mm(sign)


class MixOptimizer:
    """Wraps an optimizer; projects Stiefel gradients and retracts after."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = {}

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None or not isinstance(p, StiefelParameter):
                    continue
                self.state[id(p)] = p.data.clone()
                p.data.fill_(0)
                p.grad.data.copy_(_orthogonal_projection(p.grad.data,
                                                         self.state[id(p)]))
        self.optimizer.step()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None or not isinstance(p, StiefelParameter):
                    continue
                p.data.copy_(_retraction(p.data, self.state[id(p)]))


class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = StiefelParameter(torch.empty(input_size, output_size))
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        w = self.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.bmm(w.transpose(1, 2), torch.bmm(x, w))


class SPDTangentSpaceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = x.new(x.size(0), x.size(1), x.size(2))
        for k, xk in enumerate(x):
            u, s, _v = xk.svd()
            out[k] = u.mm(s.log().diag().mm(u.t()))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        eye = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
        grad = x.new(x.size(0), x.size(1), x.size(1))
        for k, g in enumerate(grad_output):
            u, s, _v = x[k].svd()
            g = _symmetric(g)
            dLdV = 2 * (g.mm(u.mm(s.log().diag())))
            dLdS = eye * ((1 / s).diag().mm(u.t().mm(g.mm(u))))
            P = s.unsqueeze(1).expand(-1, s.size(0))
            P = P - P.t()
            zero = P.abs() == 0
            P = 1 / P
            P[zero] = 0
            grad[k] = u.mm(_symmetric(P.t() * (u.t().mm(dLdV))) + dLdS
                           ).mm(u.t())
        return grad


class SPDTangentSpace(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        rows, cols = torch.triu_indices(input_size, input_size)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)

    def forward(self, x):
        return SPDTangentSpaceFunction.apply(x)[:, self.rows, self.cols]


class SPDRectifiedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, epsilon):
        ctx.save_for_backward(x, epsilon)
        out = x.new(x.size(0), x.size(1), x.size(2))
        for k, xk in enumerate(x):
            u, s, _v = xk.svd()
            s[s < epsilon[0]] = epsilon[0]
            out[k] = u.mm(s.diag().mm(u.t()))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, epsilon = ctx.saved_tensors
        eye = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
        grad = x.new(x.size(0), x.size(1), x.size(2))
        for k, g in enumerate(grad_output):
            g = _symmetric(g)
            u, s, _v = x[k].svd()
            keep = s > epsilon
            s_max = s.clone()
            s_max[~keep] = epsilon
            dLdV = 2 * (g.mm(u.mm(s_max.diag())))
            dLdS = eye * (keep.to(x.dtype).diag().mm(u.t().mm(g.mm(u))))
            P = s.unsqueeze(1).expand(-1, s.size(0))
            P = P - P.t()
            zero = P.abs() == 0
            P = 1 / P
            P[zero] = 0
            grad[k] = u.mm(_symmetric(P.t() * u.t().mm(dLdV)) + dLdS
                           ).mm(u.t())
        return grad, None


class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor([epsilon]))

    def forward(self, x):
        return SPDRectifiedFunction.apply(x, self.epsilon)


def _signal2spd(x):
    x = x - x.mean(dim=-1, keepdim=True)
    cov = x @ x.transpose(1, 2) / (x.shape[-1] - 1)
    cov = cov / cov.diagonal(dim1=-1, dim2=-2).sum(-1).view(-1, 1, 1)
    return cov + 1e-5 * torch.eye(cov.shape[-1], device=x.device)


class E2R(nn.Module):
    """Split the time axis into ``epochs`` segments -> one SPD matrix each."""

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def forward(self, x):
        x = x.squeeze(2)                         # (B, F, T)
        n = x.shape[-1]
        sizes = [n // self.epochs + (i < n % self.epochs)
                 for i in range(self.epochs)]
        return torch.stack([_signal2spd(part)
                            for part in torch.split(x, sizes, dim=-1)], dim=1)


class AttentionManifold(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.q_trans = SPDTransform(d_in, d_out)
        self.k_trans = SPDTransform(d_in, d_out)
        self.v_trans = SPDTransform(d_in, d_out)

    @staticmethod
    def tensor_log(t):
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    @staticmethod
    def tensor_exp(t):
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)

    def log_euclidean_distance(self, A, B):
        inner = self.tensor_log(A) - self.tensor_log(B)
        _, s, _ = torch.svd(inner @ inner.permute(0, 1, 3, 2))
        return s.sum(dim=-1)

    def log_euclidean_mean(self, weight, cov):
        bs, num_p, size = cov.shape[0], cov.shape[1], cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        out = (weight @ cov).view(bs, num_p, size, size)
        return self.tensor_exp(out)

    def forward(self, x):
        x = x.to(torch.float)
        bs, m = x.shape[0], x.shape[1]
        x = x.reshape(bs * m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)
        Q_expand = Q.repeat(1, m, 1, 1)
        K_expand = K.unsqueeze(2).repeat(1, 1, m, 1, 1).view(
            bs, m * m, self.d_out, self.d_out)
        energy = self.log_euclidean_distance(Q_expand, K_expand).view(bs, m, m)
        prob = nn.Softmax(dim=-2)(1 / (1 + torch.log(1 + energy))
                                  ).permute(0, 2, 1)
        out = self.log_euclidean_mean(prob, V)
        return out.reshape(-1, self.d_out, self.d_out), (bs, m)


class MAtt(nn.Module):
    """Official ``mAtt_bci`` (BCI-IV-2a) with task-sized conv and output."""

    def __init__(self, n_chans, n_classes, epochs=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 22, (n_chans, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(20)
        self.ract1 = E2R(epochs)
        self.att2 = AttentionManifold(20, 18)
        self.ract2 = SPDRectified()
        self.tangent = SPDTangentSpace(18)
        self.linear = nn.Linear(9 * 19 * epochs, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        x = self.Bn2(self.conv2(self.Bn1(self.conv1(x))))
        x, (bs, m) = self.att2(self.ract1(x))
        x = self.tangent(self.ract2(x))
        return self.linear(x.view(bs, -1))


# ---------------------------------------------------------------------------
# RGNN — official SymSimGCNNet (https://github.com/zhongpeixiang/RGNN) as a
# dense-adjacency PyTorch module.
# ---------------------------------------------------------------------------

DE_BANDS = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 50))
# Left/right symmetric pairs that get a global (negative) connection.
RGNN_PAIRS = (("FP1", "FP2"), ("AF3", "AF4"), ("F5", "F6"), ("FC5", "FC6"),
              ("C5", "C6"), ("CP5", "CP6"), ("P5", "P6"), ("PO5", "PO6"),
              ("O1", "O2"))


def _clean_name(name):
    return str(name).removeprefix("EEG ").split("-", 1)[0].upper()


def _rgnn_adjacency(ch_names, delta=5.0):
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = {k.upper(): v for k, v in
                 montage.get_positions()["ch_pos"].items()}
    names = [_clean_name(n) for n in ch_names]
    missing = [n for n in names if n not in positions]
    if missing:
        raise ValueError("RGNN cannot place channels: " + ", ".join(missing))
    pos = torch.tensor([positions[n] for n in names], dtype=torch.float32)
    dist_cm = torch.cdist(pos, pos) * 100
    A = torch.clamp(delta / dist_cm.pow(2).clamp_min(1e-6), max=1.0)
    index = {n: i for i, n in enumerate(names)}
    for a, b in RGNN_PAIRS:
        if a in index and b in index:
            i, j = index[a], index[b]
            A[i, j] -= 1
            A[j, i] -= 1
    return A


def differential_entropy(X, sfreq):
    """``(B, C, T)`` -> ``(B, C, 5)`` band DE = 0.5 log(2 pi e var)."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    X = np.asarray(X.detach().cpu() if torch.is_tensor(X) else X,
                   dtype=np.float64)
    feats = []
    for low, high in DE_BANDS:
        high = min(high, 0.45 * sfreq)
        sos = butter(4, [low, high], btype="bandpass", fs=sfreq,
                     output="sos")
        var = sosfiltfilt(sos, X, axis=-1).var(axis=-1)
        feats.append(0.5 * np.log(2 * np.pi * np.e * var + 1e-12))
    return torch.tensor(np.stack(feats, axis=-1), dtype=torch.float32)


class RGNN(nn.Module):

    def __init__(self, ch_names, sfreq, n_classes, K=2, hidden=64,
                 dropout=0.5):
        super().__init__()
        A = _rgnn_adjacency(ch_names)
        n = A.shape[0]
        rows, cols = torch.tril_indices(n, n)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.edge_weight = nn.Parameter(A[rows, cols].clone())
        self.register_buffer("feat_mean", torch.zeros(n, len(DE_BANDS)))
        self.register_buffer("feat_std", torch.ones(n, len(DE_BANDS)))
        self.sfreq, self.K, self.n = float(sfreq), K, n
        self.lin = nn.Linear(len(DE_BANDS), hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def adjacency(self):
        A = self.edge_weight.new_zeros(self.n, self.n)
        A[self.rows, self.cols] = self.edge_weight
        A = A + A.t() - torch.diag(A.diagonal())
        deg_inv_sqrt = A.abs().sum(dim=1).pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]

    def fit_standardization(self, X):
        de = differential_entropy(X, self.sfreq)
        self.feat_mean.copy_(de.mean(dim=0))
        self.feat_std.copy_(de.std(dim=0).clamp_min(1e-6))

    def forward(self, X):
        device = self.edge_weight.device
        x = differential_entropy(X, self.sfreq).to(device)
        x = (x - self.feat_mean) / self.feat_std      # (B, C, 5)
        A = self.adjacency()
        for _ in range(self.K):
            x = A @ x
        x = torch.relu(self.lin(x)).sum(dim=1)        # sum pool over nodes
        return self.fc(self.dropout(x))


# ---------------------------------------------------------------------------
# EEGMamba — official architecture (https://github.com/wjq-learning/EEGMamba,
# MIT, (c) 2025 Jiquan Wang) with mamba-ssm replaced by pure PyTorch.
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z=None):
        if z is not None:          # gated (Mamba-2, norm_before_gate=False)
            x = x * nn.functional.silu(z)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def _segsum(x):
    """Stable segment sum: out[..., i, j] = sum(x[..., j+1:i+1]), -inf above."""
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1)
    x = x.masked_fill(~keep, 0).cumsum(dim=-2)
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    return x.masked_fill(~keep, float("-inf"))


def ssd_scan(X, A, B, C, chunk=64):
    """Mamba-2 SSD reference scan (``ssd_minimal_discrete``).

    X: (b, l, h, p) inputs already scaled by dt; A: (b, l, h) = A * dt;
    B, C: (b, l, h, n). Returns Y: (b, l, h, p).
    """
    b, length, h, p = X.shape
    pad = (-length) % chunk
    if pad:
        X, B, C = (nn.functional.pad(t, (0, 0, 0, 0, 0, pad)) for t in (X, B, C))
        A = nn.functional.pad(A, (0, 0, 0, pad))
    c = X.shape[1] // chunk
    X, B, C = (t.reshape(b, c, chunk, *t.shape[2:]) for t in (X, B, C))
    A = A.reshape(b, c, chunk, h).permute(0, 3, 1, 2)       # (b, h, c, l)
    A_cumsum = A.cumsum(dim=-1)
    L = torch.exp(_segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(_segsum(
        nn.functional.pad(A_cumsum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states,
                         torch.exp(A_cumsum))
    Y = (Y_diag + Y_off).reshape(b, c * chunk, h, p)
    return Y[:, :length]


class Mamba2(nn.Module):
    """``mamba_ssm.modules.mamba2.Mamba2`` defaults (ngroups 1, d_conv 4,
    expand 2, gated RMSNorm), forward pass in plain PyTorch."""

    def __init__(self, d_model, d_state=64, headdim=50, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim,
                                padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1)
                       - math.log(1e-3)) + math.log(1e-3)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads)
                                            .uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        batch, length, _ = u.shape
        z, xBC, dt = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1)
        xBC = nn.functional.silu(
            self.conv1d(xBC.transpose(1, 2))[..., :length].transpose(1, 2))
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = nn.functional.softplus(dt + self.dt_bias)        # (b, l, h)
        A = -torch.exp(self.A_log.float())
        x = x.reshape(batch, length, self.nheads, self.headdim)
        B = B[:, :, None, :].expand(-1, -1, self.nheads, -1)
        C = C[:, :, None, :].expand(-1, -1, self.nheads, -1)
        y = ssd_scan(x * dt[..., None], A * dt, B, C)
        y = y + x * self.D[:, None]
        y = self.norm(y.reshape(batch, length, self.d_inner), z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """``mamba_ssm`` Block: add residual -> RMSNorm -> mixer (no MLP)."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba2(d_model)

    def forward(self, hidden, residual=None):
        residual = hidden if residual is None else hidden + residual
        residual = residual.float()                     # residual_in_fp32
        return self.mixer(self.norm(residual)), residual


class EEGMambaPatchEmbedding(nn.Module):

    def __init__(self, d_model=200):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Conv2d(d_model, d_model, (7, 7),
                                             padding=(3, 3), groups=d_model,
                                             bias=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, (1, 49), stride=(1, 25), padding=(0, 24),
                      bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model, bias=False),
                                           nn.Dropout(0.1))

    def forward(self, x):
        bz, ch, patches, size = x.shape
        time_emb = self.proj_in(x.reshape(bz, 1, ch * patches, size))
        time_emb = time_emb.permute(0, 2, 1, 3).reshape(bz, ch, patches,
                                                        self.d_model)
        spectral = torch.fft.rfft(x, dim=-1, norm="forward").abs()
        emb = time_emb + self.spectral_proj(spectral)
        pos = self.positional_encoding(emb.permute(0, 3, 1, 2))
        return emb + pos.permute(0, 2, 3, 1)


class EEGMamba(nn.Module):
    """Official EEGMamba backbone + ``all_patch_reps`` classifier."""

    PATCH = 200
    SFREQ = 200

    def __init__(self, n_chans, n_times, sfreq, n_classes, n_layer=12,
                 d_model=200, dropout=0.1):
        super().__init__()
        self.sfreq = float(sfreq)
        self.n_times = (round(n_times * self.SFREQ / self.sfreq)
                        // self.PATCH) * self.PATCH
        if self.n_times == 0:
            raise ValueError("EEGMamba needs at least one second of EEG")
        n_patches = self.n_times // self.PATCH
        self.patch_embedding = EEGMambaPatchEmbedding(d_model)
        self.layers = nn.ModuleList(MambaBlock(d_model)
                                    for _ in range(n_layer))
        self.norm_f = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_chans * n_patches * d_model, n_patches * d_model),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(n_patches * d_model, d_model),
            nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")

    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        target = round(X.shape[-1] * self.SFREQ / self.sfreq)
        if target != X.shape[-1]:
            X = nn.functional.interpolate(X, size=target, mode="linear",
                                          align_corners=False)
        X = X[..., :self.n_times] / 100.0
        bz, ch = X.shape[:2]
        X = X.reshape(bz, ch, -1, self.PATCH)                # (B, C, P, 200)
        n_patches = X.shape[2]
        hidden = self.patch_embedding(X).reshape(bz, ch * n_patches, -1)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
            hidden, residual = hidden.flip(1), residual.flip(1)
        hidden = self.norm_f(hidden + residual)
        return self.classifier(hidden.reshape(bz, ch, n_patches, -1))


# ---------------------------------------------------------------------------
# Dual-TSST — reimplemented from arXiv 2409.03251 (no public code).
# ---------------------------------------------------------------------------

class MorletAmplitude(nn.Module):
    """Morlet CWT amplitude ``(B, C, T) -> (B, C, F, T)`` as a GPU conv.

    Wavelets come from ``mne.time_frequency.morlet`` (n_cycles = freq / 2,
    zero mean), as with MNE's ``tfr_array_morlet``.
    """

    def __init__(self, sfreq, freqs):
        super().__init__()
        import numpy as np
        from mne.time_frequency import morlet
        waves = morlet(sfreq, freqs, n_cycles=np.asarray(freqs) / 2,
                       zero_mean=True)
        size = max(len(w) for w in waves)
        size += 1 - size % 2
        kernel = np.zeros((len(waves), size), dtype=np.complex128)
        for i, w in enumerate(waves):
            start = (size - len(w)) // 2
            kernel[i, start:start + len(w)] = w[::-1]   # convolution
        k = torch.tensor(np.concatenate([kernel.real, kernel.imag]),
                         dtype=torch.float32)
        self.register_buffer("kernel", k[:, None, :])  # (2F, 1, K)
        self.n_freqs = len(waves)

    def forward(self, X):
        B, C, T = X.shape
        out = nn.functional.conv1d(X.reshape(B * C, 1, T), self.kernel,
                                   padding=self.kernel.shape[-1] // 2)
        real, imag = out[:, :self.n_freqs], out[:, self.n_freqs:]
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-12).reshape(
            B, C, self.n_freqs, T)


def _tsst_stream(in_ch, depth, t_kernel, pool, stride, d1=40, d2=120):
    """time conv -> BN -> depthwise conv over ``depth`` -> BN -> ELU ->
    avg-pool -> pointwise conv; ``(B, in_ch, depth, T) -> (B, d2, 1, T')``."""
    return nn.Sequential(
        nn.Conv2d(in_ch, d1, (1, t_kernel)),
        nn.BatchNorm2d(d1),
        nn.Conv2d(d1, d1, (depth, 1), groups=d1),
        nn.BatchNorm2d(d1),
        nn.ELU(),
        nn.AvgPool2d((1, pool), stride=(1, stride)),
        nn.Conv2d(d1, d2, (1, 1)),
    )


class DualTSST(nn.Module):

    SFREQ = 250.0

    def __init__(self, n_chans, n_times, sfreq, n_classes, d2=120, heads=10,
                 layers=4, ff=480, dropout=0.1, hidden=64):
        super().__init__()
        self.sfreq = float(sfreq)
        self.n_times = round(n_times * self.SFREQ / self.sfreq)
        freqs = list(range(1, 41))
        self.cwt = MorletAmplitude(self.SFREQ, freqs)
        F = len(freqs)
        self.branch1 = _tsst_stream(1, n_chans, 30, 120, 12, d2=d2)
        self.branch2a = _tsst_stream(n_chans, F, 125, 64, 32, d2=d2)
        self.branch2b = _tsst_stream(F, n_chans, 125, 64, 32, d2=d2)
        t1 = (self.n_times - 29 - 120) // 12 + 1
        t2 = (self.n_times - 124 - 64) // 32 + 1
        self.pos = nn.Parameter(torch.randn(1, t1 + 2 * t2, d2) * 0.02)
        layer = nn.TransformerEncoderLayer(d2, heads, ff, dropout,
                                           batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
        self.mlp = nn.Sequential(nn.Linear(d2, hidden), nn.ELU(),
                                 nn.Linear(hidden, n_classes))

    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.shape[-1] != self.n_times:
            X = nn.functional.interpolate(X, size=self.n_times,
                                          mode="linear", align_corners=False)
        W = self.cwt(X)                                    # (B, C, F, T)
        tokens = [
            self.branch1(X.unsqueeze(1)),                  # (B, d2, 1, T')
            self.branch2a(W),                              # in: ch, F x T
            self.branch2b(W.transpose(1, 2)),              # in: F, ch x T
        ]
        z = torch.cat([t.squeeze(2).transpose(1, 2) for t in tokens], dim=1)
        z = self.encoder(z + self.pos)
        return self.mlp(z.mean(dim=1))


# ---------------------------------------------------------------------------

# Used as-is from braindecode with its default hyperparameters.
BRAINDECODE_MODELS = ("AttentionBaseNet", "EEGNet", "ShallowFBCSPNet",
                      "Deep4Net", "EEGTCNet", "EEGConformer")


def _build_network(name, meta):
    kwargs = dict(n_chans=meta["n_chans"], n_outputs=meta["n_classes"],
                  n_times=meta["n_times"], sfreq=meta["sfreq"])
    if name in BRAINDECODE_MODELS:
        import braindecode.models as models
        return getattr(models, name)(**kwargs)
    if name == "MAtt":
        return MAtt(meta["n_chans"], meta["n_classes"])
    if name == "RGNN":
        return RGNN(meta["ch_names"], meta["sfreq"], meta["n_classes"])
    if name == "DualTSST":
        return DualTSST(meta["n_chans"], meta["n_times"], meta["sfreq"],
                        meta["n_classes"])
    if name == "EEGMamba":
        return EEGMamba(meta["n_chans"], meta["n_times"], meta["sfreq"],
                        meta["n_classes"])
    raise ValueError(f"Unknown scratch model: {name}")


def _optimizer(name, net):
    if name == "MAtt":   # official: Adam(wd=0.1) + Stiefel MixOptimizer
        return MixOptimizer(torch.optim.Adam(net.parameters(), lr=5e-4,
                                             weight_decay=1e-1))
    return torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)


# Models whose official code runs on CPU (per-sample SVD loops).
CPU_ONLY = {"MAtt"}
# Models that take raw microvolts (RGNN's band DE needs absolute power).
RAW_INPUT = {"RGNN", "EEGMamba"}


def _wandb():
    if os.environ.get("WANDB_MODE", "online").lower() == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def _standardize(X):
    X = torch.as_tensor(X, dtype=torch.float32)
    mean = X.mean(dim=-1, keepdim=True)
    std = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (X - mean) / std


class ScratchModel:

    wandb_run_id = None

    def __init__(self, net, device, standardize=True):
        self.net = net.to(device)
        self.device = device
        self.standardize = standardize

    def prepare(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        return _standardize(X) if self.standardize else X

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        return self.net(self.prepare(X).to(self.device)).argmax(dim=1)

    def log_evaluation(self, metrics):
        wandb = _wandb()
        if wandb is None or self.wandb_run_id is None:
            return
        run = wandb.run
        if run is None or run.id != self.wandb_run_id:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
                id=self.wandb_run_id, resume="allow",
                reinit="finish_previous",
            )
        run.define_metric("evaluation_step")
        run.define_metric("evaluation/*", step_metric="evaluation_step")
        run.log({
            "evaluation_step": 0,
            **{f"evaluation/{key}": value for key, value in metrics.items()},
        })
        run.finish()


class Solver(CompetSolver):

    name = "Scratch"

    requirements = ["pip::braindecode", "pip::wandb"]
    parameters = {"model": ["AttentionBaseNet", "MAtt", "RGNN", "EEGMamba",
                            "DualTSST", "EEGNet", "ShallowFBCSPNet",
                            "Deep4Net", "EEGTCNet", "EEGConformer"]}

    def set_objective(self, train_loader, **meta):
        self.name = self.run_label(self.model, "scratch", "shotall")
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        if self.model in CPU_ONLY:
            self.device = "cpu"
        model = ScratchModel(_build_network(self.model, meta), self.device,
                             standardize=self.model not in RAW_INPUT)
        weights = meta["submission_dir"] / "weights.pt"
        if self.train_loader is None:
            if not weights.is_file():
                raise FileNotFoundError(
                    f"{self.model} inference needs weights.pt. Train with "
                    "training=True and upload the exported submission folder."
                )
            print(f"[loading] {weights} into {self.name}")
            model.net.load_state_dict(torch.load(
                weights, map_location=self.device, weights_only=True,
            ))
        return model

    def fit(self, model, train_loader, epochs=20, batch_size=16):
        signals, labels = [], []
        for X, y, _info in train_loader:
            signals.append(model.prepare(X.detach().cpu()))
            labels.append(y.detach().reshape(-1).cpu().long())
        signals, labels = torch.cat(signals), torch.cat(labels)

        if hasattr(model.net, "fit_standardization"):
            model.net.fit_standardization(signals)
        run = self._start_wandb(model.net, len(labels))
        net = model.net
        optimizer = _optimizer(self.model, net)
        loss_fn = nn.CrossEntropyLoss()
        try:
            for epoch in range(epochs):
                net.train()
                total_loss, correct = 0.0, 0
                for indices in torch.randperm(len(signals)).split(batch_size):
                    X = signals[indices].to(self.device)
                    y = labels[indices].to(self.device)
                    optimizer.zero_grad()
                    logits = net(X)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(indices)
                    correct += (logits.argmax(dim=1) == y).sum().item()
                if run is not None:
                    run.log({
                        "epoch": epoch,
                        "train/loss": total_loss / len(signals),
                        "train/accuracy": correct / len(signals),
                    })
        except BaseException:
            if run is not None:
                run.finish(exit_code=1)
            raise
        if run is not None:
            model.wandb_run_id = run.id

    def _start_wandb(self, net, n_train):
        wandb = _wandb()
        if wandb is None:
            return None
        context = getattr(self, "_run_context", None)
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
            name=self.name,
            config={
                "model": self.model,
                "strategy": "scratch",
                "n_shot_per_class": "all",
                "n_train_windows": n_train,
                "pretrained": False,
                "n_params": sum(p.numel() for p in net.parameters()),
                "n_classes": self.meta["n_classes"],
                "n_chans": self.meta["n_chans"],
                "n_times": self.meta["n_times"],
                "sfreq": self.meta["sfreq"],
                "seed": getattr(context, "base_seed", None),
            },
            reinit="finish_previous",
        )

    def save_model(self, model, path):
        torch.save(model.net.state_dict(), path / "weights.pt")
