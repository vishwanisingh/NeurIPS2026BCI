"""CBraMod — pretrained criss-cross EEG transformer (braindecode).

One configurable solver mirroring the LaBraM / REVE experiments:

- ``strategy=probe``      : frozen pretrained encoder + ridge classifier on
                            mean-pooled patch embeddings.
- ``strategy=last_block`` : fine-tune the final transformer layer + head.
- ``strategy=full``       : fine-tune the whole pretrained encoder + head.
- ``strategy=scratch``    : same architecture, random init, trained fully.

``n_shot=0`` uses every labeled training window; ``n_shot=5|10`` samples that
many windows per class (seeded by the Benchopt run seed).

Input handling follows the CBraMod paper: windows are resampled to 200 Hz
(4 s -> 800 samples = four 1-s patches) and microvolts are divided by 100.
The pretrained reconstruction head (``proj_out``) is replaced by identity, so
features are the encoder's ``(B, C, P, 200)`` patch embeddings.

The exported submission ships the complete encoder state (``weights.pt``,
~20 MB) — CBraMod is not staged on competition workers, so inference never
touches the Hugging Face Hub. The probe also ships ``weights.joblib``.

If W&B is installed and logged in, each training run logs per-epoch training
loss/accuracy and the held-out benchmark metrics to ``neurips-eeg-2026-bci``
(override with ``WANDB_PROJECT``; ``WANDB_MODE=disabled`` turns it off). The
W&B run name matches the output folder, e.g. ``CBraMod-full-shot5-seed42``.
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

CBRAMOD_SFREQ = 200
PATCH_SIZE = 200
HUB_ID = "braindecode/cbramod-pretrained"
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
    from braindecode.models import CBraMod
    kwargs = dict(n_chans=n_chans, n_times=n_times, n_outputs=1,
                  sfreq=CBRAMOD_SFREQ, return_encoder_output=True)
    if pretrained:
        backbone = CBraMod.from_pretrained(HUB_ID, **kwargs)
    else:
        backbone = CBraMod(**kwargs)
    backbone.proj_out = nn.Identity()
    backbone.final_layer = nn.Identity()
    return backbone


class CBraModClassifier(nn.Module):
    """CBraMod backbone + mean-pooled LayerNorm/Linear head."""

    def __init__(self, meta, pretrained):
        super().__init__()
        self.sfreq = float(meta["sfreq"])
        self.n_times = max(
            round(meta["n_times"] * CBRAMOD_SFREQ / self.sfreq), 1
        )
        if self.n_times < PATCH_SIZE or self.n_times % PATCH_SIZE:
            # CBraMod splits time into whole 1-s patches; trim to a multiple.
            self.n_times = (self.n_times // PATCH_SIZE) * PATCH_SIZE
            if self.n_times == 0:
                raise ValueError("CBraMod needs at least one second of EEG")
        self.backbone = _build_backbone(
            meta["n_chans"], self.n_times, pretrained
        )
        dim = PATCH_SIZE
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, meta["n_classes"])
        )

    def features(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        X = X.to(next(self.parameters()).device)
        target = round(X.shape[-1] * CBRAMOD_SFREQ / self.sfreq)
        if target != X.shape[-1]:
            X = F.interpolate(X, size=target, mode="linear",
                              align_corners=False)
        X = X[..., :self.n_times] / 100.0
        feats = self.backbone(X)                # (B, C, P, 200)
        return feats.mean(dim=(1, 2))           # (B, 200)

    def forward(self, X):
        return self.head(self.features(X))


class _Probe(_LogsEvaluation):
    """Frozen CBraMod features + scikit-learn ridge head."""

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

    name = "CBraMod"

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
            "CBraMod", self.strategy, f"shot{shot_label}"
        )
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        training = self.train_loader is not None
        if not training and not weights.is_file():
            raise FileNotFoundError(
                "CBraMod inference needs weights.pt. Train with "
                "training=True and upload the exported submission folder."
            )
        pretrained = training and self.strategy != "scratch"
        model = CBraModClassifier(meta, pretrained=pretrained)
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
            backbone.encoder.layers[-1].requires_grad_(True)
        elif self.strategy in ("full", "scratch"):
            backbone.requires_grad_(True)
        else:
            raise ValueError(f"Unknown CBraMod strategy: {self.strategy}")
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
                "model": "CBraMod",
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
