"""BIOT — pretrained biosignal transformer (braindecode).

One configurable solver mirroring the REVE / LaBraM / CBraMod experiments:

- ``strategy=probe``      : frozen pretrained encoder + ridge classifier on
                            its pooled embedding.
- ``strategy=last_block`` : fine-tune the final transformer layer + head.
- ``strategy=full``       : fine-tune the whole pretrained encoder + head.
- ``strategy=scratch``    : same architecture, random init, trained fully.

``n_shot=0`` uses every labeled training window; ``n_shot=5|10`` samples that
many windows per class (seeded by the Benchopt run seed).

Weights come from ``braindecode/biot-pretrained-six-datasets-18chs``. That
checkpoint has learned tokens for 18 (undocumented) pretraining channels, so
this solver keeps all of the task's channels and gives each a freshly
initialized channel token; the STFT patch embedding, linear-attention
transformer and positional encoding are pretrained. Windows are resampled to
200 Hz and each channel is scaled by its 95th-percentile absolute amplitude,
as in BIOT's preprocessing.

The exported submission ships the complete model state (``weights.pt``,
~13 MB) — BIOT is not staged on competition workers, so inference never
touches the Hugging Face Hub. The probe also ships ``weights.joblib``.

If W&B is installed and logged in, each training run logs per-epoch training
loss/accuracy and the held-out benchmark metrics to ``neurips-eeg-2026-bci``
(override with ``WANDB_PROJECT``; ``WANDB_MODE=disabled`` turns it off). The
W&B run name matches the output folder, e.g. ``BIOT-full-shot5-seed42``.
"""

import os

import joblib
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver

BIOT_SFREQ = 200
HUB_ID = "braindecode/biot-pretrained-six-datasets-18chs"
WANDB_PROJECT = "neurips-eeg-2026-bci"


def _wandb():
    """The ``wandb`` module, or None when it is missing or disabled."""
    if os.environ.get("WANDB_MODE", "online").lower() == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb


class _LogsEvaluation:
    """Objective hook: append held-out metrics to the training W&B run."""

    wandb_run_id = None

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


def _build_backbone(n_chans, n_times, pretrained):
    """BIOT for ``n_chans`` channels; optionally transplant pretrained weights.

    The checkpoint's channel tokens belong to its 18 pretraining channels, so
    they (and its classifier) are left at fresh initialization; the patch
    embedding, transformer and positional encoding are pretrained.
    """
    from braindecode.models import BIOT
    backbone = BIOT(n_chans=n_chans, n_times=n_times, n_outputs=1,
                    sfreq=BIOT_SFREQ)
    if pretrained:
        source = BIOT.from_pretrained(HUB_ID).state_dict()
        backbone.load_state_dict({
            key: value for key, value in source.items()
            if not key.startswith(("encoder.channel_tokens", "final_layer"))
        }, strict=False)
    backbone.final_layer = nn.Identity()
    return backbone


class BIOTClassifier(nn.Module):
    """BIOT backbone + LayerNorm/Linear head on its pooled embedding."""

    def __init__(self, meta, pretrained):
        super().__init__()
        self.sfreq = float(meta["sfreq"])
        self.n_times = max(
            round(meta["n_times"] * BIOT_SFREQ / self.sfreq), 1
        )
        self.backbone = _build_backbone(
            meta["n_chans"], self.n_times, pretrained
        )
        dim = self.backbone.encoder.channel_tokens.embedding_dim
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, meta["n_classes"])
        )

    def features(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        X = X.to(next(self.parameters()).device)
        if X.shape[-1] != self.n_times:
            X = F.interpolate(X, size=self.n_times, mode="linear",
                              align_corners=False)
        # BIOT's preprocessing: scale each channel by its 95th percentile
        # absolute amplitude.
        scale = torch.quantile(X.abs(), 0.95, dim=-1, keepdim=True)
        X = X / (scale + 1e-8)
        return self.backbone(X, return_features=True)["features"]  # (B, D)

    def forward(self, X):
        return self.head(self.features(X))


class _Probe(_LogsEvaluation):
    """Frozen BIOT features + scikit-learn ridge head."""

    def __init__(self, model, head=None):
        self.model = model
        self.head = head or make_pipeline(StandardScaler(), RidgeClassifier())

    @torch.inference_mode()
    def _embed(self, X, batch_size=64):
        self.model.eval()
        return torch.cat([
            self.model.features(X[i:i + batch_size]).cpu()
            for i in range(0, len(X), batch_size)
        ]).numpy()

    def fit(self, X, y):
        self.head.fit(self._embed(X), y.numpy())

    def predict(self, X):
        return self.head.predict(self._embed(X))


class _FineTuned(_LogsEvaluation):
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def predict(self, X):
        self.model.eval()
        return self.model(X).argmax(dim=1)


class Solver(CompetSolver):

    name = "BIOT"

    requirements = ["pip::braindecode", "pip::huggingface_hub", "pip::wandb"]
    parameters = {
        "strategy": ["probe", "last_block", "full", "scratch"],
        "n_shot": [0, 5, 10],
    }

    def skip(self, train_loader, **meta):
        if self.strategy == "scratch" and self.n_shot:
            return True, "scratch is only run with all training data"
        return False, None

    def set_objective(self, train_loader, **meta):
        shot_label = "all" if self.n_shot == 0 else str(self.n_shot)
        self.name = self.run_label(
            "BIOT", self.strategy, f"shot{shot_label}"
        )
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        training = self.train_loader is not None
        if not training and not weights.is_file():
            raise FileNotFoundError(
                "BIOT inference needs weights.pt. Train with "
                "training=True and upload the exported submission folder."
            )
        pretrained = training and self.strategy != "scratch"
        model = BIOTClassifier(meta, pretrained=pretrained)
        if not training:
            print(f"[loading] {weights} into {self.name}")
            model.load_state_dict(torch.load(
                weights, map_location="cpu", weights_only=True,
            ))
        model.to(self.device)

        backbone = model.backbone
        if self.strategy == "probe":
            model.requires_grad_(False)
            head = None
            if not training:
                head = joblib.load(meta["submission_dir"] / "weights.joblib")
            return _Probe(model, head)
        if self.strategy == "last_block":
            backbone.requires_grad_(False)
            backbone.encoder.transformer.layers.layers[-1].requires_grad_(
                True
            )
        elif self.strategy in ("full", "scratch"):
            backbone.requires_grad_(True)
        else:
            raise ValueError(f"Unknown BIOT strategy: {self.strategy}")
        model.head.requires_grad_(True)
        return _FineTuned(model)

    def fit(self, model, train_loader):
        signals, labels = self._collect(train_loader)
        if self.n_shot:
            signals, labels = self._sample_shots(signals, labels)
        run = self._start_wandb(n_train=len(labels))
        try:
            if self.strategy == "probe":
                model.fit(signals, labels)
                if run is not None:
                    preds = torch.as_tensor(model.predict(signals))
                    run.log({"train/accuracy":
                             (preds == labels).float().mean().item()})
            else:
                self._fine_tune(model.model, signals, labels, run)
        except BaseException:
            if run is not None:
                run.finish(exit_code=1)
            raise
        if run is not None:
            # The objective's log_evaluation adds test metrics, then finishes.
            model.wandb_run_id = run.id

    def _start_wandb(self, n_train):
        wandb = _wandb()
        if wandb is None:
            return None
        context = getattr(self, "_run_context", None)
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
            name=self.name,
            config={
                "model": "BIOT",
                "strategy": self.strategy,
                "n_shot_per_class": self.n_shot or "all",
                "n_train_windows": n_train,
                "pretrained": self.strategy != "scratch",
                "n_classes": self.meta["n_classes"],
                "n_chans": self.meta["n_chans"],
                "n_times": self.meta["n_times"],
                "sfreq": self.meta["sfreq"],
                "seed": getattr(context, "base_seed", None),
            },
            reinit="finish_previous",
        )

    @staticmethod
    def _collect(train_loader):
        signals, labels = [], []
        for X, y, _info in train_loader:
            signals.append(X.detach().cpu())
            labels.append(y.detach().reshape(-1).cpu().long())
        return torch.cat(signals), torch.cat(labels)

    def _sample_shots(self, signals, labels):
        generator = torch.Generator().manual_seed(torch.initial_seed())
        selected = []
        for label in torch.unique(labels, sorted=True):
            indices = torch.where(labels == label)[0]
            if len(indices) < self.n_shot:
                raise ValueError(
                    f"Class {int(label)} has {len(indices)} training windows; "
                    f"cannot sample {self.n_shot} shots"
                )
            order = torch.randperm(len(indices), generator=generator)
            selected.append(indices[order[:self.n_shot]])
        selected = torch.cat(selected)
        selected = selected[torch.randperm(len(selected), generator=generator)]
        return signals[selected], labels[selected]

    def _fine_tune(self, model, signals, labels, run=None, epochs=20,
                   batch_size=16):
        backbone_lr = 5e-4 if self.strategy == "scratch" else 5e-5
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.backbone.parameters()
                            if p.requires_grad], "lr": backbone_lr},
                {"params": model.head.parameters(), "lr": 5e-4},
            ],
            weight_decay=0.05,
        )
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(signals))
            total_loss, correct = 0.0, 0
            for indices in order.split(batch_size):
                X = signals[indices].to(self.device)
                y = labels[indices].to(self.device)
                optimizer.zero_grad()
                logits = model(X)
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

    def save_model(self, model, path):
        torch.save(model.model.state_dict(), path / "weights.pt")
        if self.strategy == "probe":
            joblib.dump(model.head, path / "weights.joblib")
