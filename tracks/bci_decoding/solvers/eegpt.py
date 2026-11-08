"""EEGPT — pretrained dual-SSL EEG transformer (braindecode).

One configurable solver mirroring the REVE / LaBraM / CBraMod / BIOT
experiments:

- ``strategy=probe``      : frozen pretrained encoder + ridge classifier on
                            its time-pooled summary tokens.
- ``strategy=last_block`` : fine-tune the final transformer block (and the
                            encoder's output norm) + head.
- ``strategy=full``       : fine-tune the whole pretrained encoder + head.
- ``strategy=scratch``    : same architecture, random init, trained fully.

``n_shot=0`` uses every labeled training window; ``n_shot=5|10`` samples that
many windows per class (seeded by the Benchopt run seed).

Weights come from ``braindecode/eegpt-pretrained`` (4-s windows at 250 Hz).
EEGPT has a learned embedding for each of 62 named 10-10 channels, so the
task's channels are matched by name (``T3/T4/T5/T6`` mapped to
``T7/T8/P7/P8``) and channels outside that vocabulary are dropped; no
channel projection is learned. Windows are resampled to 250 Hz and converted
from microvolts to millivolts. Features are the encoder's 4 summary tokens,
averaged over time patches (4 x 512 = 2048 dims).

The exported submission ships the complete model state (``weights.pt``,
~100 MB) — EEGPT is not staged on competition workers, so inference never
touches the Hugging Face Hub. The probe also ships ``weights.joblib``.

If W&B is installed and logged in, each training run logs per-epoch training
loss/accuracy and the held-out benchmark metrics to ``neurips-eeg-2026-bci``
(override with ``WANDB_PROJECT``; ``WANDB_MODE=disabled`` turns it off). The
W&B run name matches the output folder, e.g. ``EEGPT-full-shot5-seed42``.
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

EEGPT_SFREQ = 250
HUB_ID = "braindecode/eegpt-pretrained"
# Older 10-20 names -> the 10-10 names in EEGPT's channel vocabulary.
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
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


class _Probe(_LogsEvaluation):
    """Frozen EEGPT features + scikit-learn ridge head."""

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

    name = "EEGPT"
    micro_batch_size = 4

    requirements = ["pip::braindecode", "pip::huggingface_hub", "pip::safetensors", "pip::wandb"]
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
            "EEGPT", self.strategy, f"shot{shot_label}"
        )
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        training = self.train_loader is not None
        if not training and not weights.is_file():
            raise FileNotFoundError(
                "EEGPT inference needs weights.pt. Train with "
                "training=True and upload the exported submission folder."
            )
        pretrained = training and self.strategy != "scratch"
        model = EEGPTClassifier(meta, pretrained=pretrained)
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
            backbone.target_encoder.blocks[-1].requires_grad_(True)
            backbone.target_encoder.norm.requires_grad_(True)
        elif self.strategy in ("full", "scratch"):
            backbone.requires_grad_(True)
        else:
            raise ValueError(f"Unknown EEGPT strategy: {self.strategy}")
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
                "model": "EEGPT",
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
                optimizer.zero_grad()
                # EEGPT on 41 channels is ~1.3k tokens per window; a batch
                # of 16 does not fit on a 6 GB GPU, so accumulate gradients
                # over micro-batches (no batch norm -> identical gradients).
                for micro in indices.split(self.micro_batch_size):
                    X = signals[micro].to(self.device)
                    y = labels[micro].to(self.device)
                    logits = model(X)
                    loss = loss_fn(logits, y)
                    (loss * len(micro) / len(indices)).backward()
                    total_loss += loss.item() * len(micro)
                    correct += (logits.argmax(dim=1) == y).sum().item()
                optimizer.step()
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
