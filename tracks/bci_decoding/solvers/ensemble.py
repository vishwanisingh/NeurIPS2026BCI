"""Ensemble of the two best BCI-decoding models: MAtt + frozen EEGPT probe.

Both members were trained earlier on all labeled PROTEUS windows (seed 42):

- ``MAtt-scratch-shotall-seed42``  — manifold-attention network (CPU);
- ``EEGPT-probe-shotall-seed42``   — frozen EEGPT encoder + ridge head.

Prediction is soft voting with equal weights, fixed a priori (not tuned on
the evaluation sessions):

- MAtt: ``softmax(logits)``;
- EEGPT probe: the one-vs-rest ridge scores ``s`` regress targets in
  {-1, +1}, so ``(s + 1) / 2`` estimates each class probability; it is
  clipped to ``[1e-3, 1]`` and renormalized.

The averaged probabilities give the class. No training happens here: a
``training=True`` run loads both members from their output folders (override
with ``ENSEMBLE_MATT_DIR`` / ``ENSEMBLE_EEGPT_DIR``) and exports one
submission holding ``matt.pt``, ``eegpt.pt`` and ``eegpt_head.joblib``.

The member code below is copied from ``scratch_nets.py`` (MAtt) and
``eegpt.py``, because a submission must be a single Python file.
"""

import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmark_utils.base_solver import CompetSolver

EEGPT_SFREQ = 250
HUB_ID = "braindecode/eegpt-pretrained"
# Older 10-20 names -> the 10-10 names in EEGPT's channel vocabulary.
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
MATT_DIR = "MAtt-scratch-shotall-seed42"
EEGPT_DIR = "EEGPT-probe-shotall-seed42"

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


def _channel_names(ch_names):
    """Normalize channel names and keep those EEGPT has embeddings for.

    Returns ``(indices, names)``: input-channel indices to keep and their
    EEGPT names. Channels outside EEGPT's 62-channel vocabulary are dropped.
    """
    from braindecode.models.eegpt import EEGPT_CHANNELS
    known = set(EEGPT_CHANNELS)
    indices, names = [], []
    for i, name in enumerate(ch_names or []):
        name = str(name).removeprefix("EEG ").split("-", 1)[0].upper()
        name = ALIASES.get(name, name)
        if name in known:
            indices.append(i)
            names.append(name)
    if not names:
        raise ValueError(
            "EEGPT found none of its 62 channel names in: "
            + ", ".join(map(str, ch_names or []))
        )
    return indices, names


def _build_backbone(names, n_times, pretrained):
    """EEGPT on the named channels; optionally load pretrained encoder.

    The checkpoint's ``chans_id`` buffer lists its own 62 channels, so only
    the ``target_encoder`` weights (incl. every channel's embedding) load.
    """
    from braindecode.models import EEGPT
    backbone = EEGPT(
        n_chans=len(names), chs_info=[{"ch_name": n} for n in names],
        n_times=n_times, sfreq=EEGPT_SFREQ, n_outputs=1,
        chan_proj_type="none", return_encoder_output=True,
    )
    if pretrained:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        source = load_file(hf_hub_download(HUB_ID, "model.safetensors"))
        backbone.load_state_dict({
            key: value for key, value in source.items()
            if key.startswith("target_encoder.")
        }, strict=False)
    return backbone


class EEGPTClassifier(nn.Module):
    """EEGPT encoder + LayerNorm/Linear head on time-pooled summary tokens."""

    def __init__(self, meta, pretrained):
        super().__init__()
        self.sfreq = float(meta["sfreq"])
        self.n_times = max(
            round(meta["n_times"] * EEGPT_SFREQ / self.sfreq), 1
        )
        indices, names = _channel_names(meta.get("ch_names"))
        dropped = len(meta.get("ch_names") or []) - len(names)
        print(f"[EEGPT] using {len(names)} channels, dropped {dropped}")
        self.register_buffer("channels", torch.tensor(indices))
        self.backbone = _build_backbone(names, self.n_times, pretrained)
        dim = self.backbone.embed_num * self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, meta["n_classes"])
        )

    def features(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        X = X.to(self.channels.device)[:, self.channels]
        if X.shape[-1] != self.n_times:
            X = F.interpolate(X, size=self.n_times, mode="linear",
                              align_corners=False)
        X = X / 1000.0                          # microvolts -> millivolts
        z = self.backbone(X)                    # (B, patches, embed_num, D)
        return z.mean(dim=1).flatten(1)         # (B, embed_num * D)

    def forward(self, X):
        return self.head(self.features(X))


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def _standardize(X):
    """Per-window, per-channel z-score (MAtt's training input)."""
    mean = X.mean(dim=-1, keepdim=True)
    std = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (X - mean) / std


def ridge_probabilities(scores, eps=1e-3):
    """One-vs-rest ridge scores (targets -1/+1) -> class probabilities."""
    p = np.clip((np.asarray(scores, dtype=np.float64) + 1.0) / 2.0, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


class Ensemble:
    """Equal-weight soft vote of MAtt (CPU) and the EEGPT probe."""

    def __init__(self, matt, eegpt, head):
        self.matt = matt.cpu().eval()
        self.eegpt = eegpt.eval()
        self.head = head

    @torch.inference_mode()
    def probabilities(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        p_matt = torch.softmax(
            self.matt(_standardize(X.cpu())), dim=1).numpy()
        feats = self.eegpt.features(X).cpu().numpy()
        scores = self.head.decision_function(feats)
        p_eegpt = np.zeros_like(p_matt)
        p_eegpt[:, self.head.classes_.astype(int)] = ridge_probabilities(
            scores)
        return (p_matt + p_eegpt) / 2.0

    def predict(self, X):
        return torch.as_tensor(self.probabilities(X).argmax(axis=1))


class Solver(CompetSolver):

    name = "Ensemble-MAtt-EEGPT"

    requirements = ["pip::braindecode"]

    def set_objective(self, train_loader, **meta):
        self.name = self.run_label("Ensemble-MAtt-EEGPT", "softvote",
                                   "shotall")
        super().set_objective(train_loader, **meta)

    def _member_dirs(self, meta):
        if self.train_loader is None:
            folder = meta["submission_dir"]
            return folder, folder
        outputs = Path(meta["submission_dir"]).parent
        return (Path(os.environ.get("ENSEMBLE_MATT_DIR", outputs / MATT_DIR)),
                Path(os.environ.get("ENSEMBLE_EEGPT_DIR",
                                    outputs / EEGPT_DIR)))

    def load_model(self, meta):
        matt_dir, eegpt_dir = self._member_dirs(meta)
        training = self.train_loader is not None
        names = (("weights.pt", "weights.pt", "weights.joblib") if training
                 else ("matt.pt", "eegpt.pt", "eegpt_head.joblib"))
        paths = (matt_dir / names[0], eegpt_dir / names[1],
                 eegpt_dir / names[2])
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Ensemble needs {path}")
        self._sources = paths
        print(f"[loading] {', '.join(map(str, paths))} into {self.name}")

        matt = MAtt(meta["n_chans"], meta["n_classes"])
        matt.load_state_dict(torch.load(paths[0], map_location="cpu",
                                        weights_only=True))
        eegpt = EEGPTClassifier(meta, pretrained=False)
        eegpt.load_state_dict(torch.load(paths[1], map_location="cpu",
                                         weights_only=True))
        eegpt.requires_grad_(False)
        return Ensemble(matt, eegpt.to(self.device), joblib.load(paths[2]))

    def fit(self, model, train_loader):
        """Both members are already trained; nothing to fit."""

    def save_model(self, model, path):
        for source, name in zip(self._sources,
                                ("matt.pt", "eegpt.pt", "eegpt_head.joblib")):
            shutil.copyfile(source, path / name)
