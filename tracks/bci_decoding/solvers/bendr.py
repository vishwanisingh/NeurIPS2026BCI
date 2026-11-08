"""BENDR — pretrained conv-encoder + transformer EEG model (braindecode).

One configurable solver mirroring the REVE / LaBraM / CBraMod / BIOT / EEGPT
experiments:

- ``strategy=probe``      : frozen pretrained model + ridge classifier on the
                            contextualizer's start-token output.
- ``strategy=last_block`` : fine-tune the last transformer layer (plus the
                            contextualizer's norm and output conv) + head.
- ``strategy=full``       : fine-tune the whole pretrained model + head.
- ``strategy=scratch``    : same architecture, random init, trained fully.

``n_shot=0`` uses every labeled training window; ``n_shot=5|10`` samples that
many windows per class (seeded by the Benchopt run seed).

Weights come from ``braindecode/braindecode-bendr``. BENDR was pretrained on
a fixed 20-channel input: 19 10-20 channels in DN3's order plus ``SCALE``, a
relative-amplitude channel. This solver picks those 19 channels by name
(``T5/T6`` = ``P7/P8``); any the task lacks are spherical-spline
interpolated from all task channels (MNE, standard 10-05 positions). Each
window is resampled to 256 Hz and min-max scaled to [-1, 1] over all
channels; ``SCALE`` is the window's peak-to-peak amplitude relative to the
largest one among the training windows, mapped to [-1, 1]. That scaling
follows the DN3 Deep1010 recipe as described, not a verified port of its code.

BENDR is ~157M parameters, so the exported ``weights.pt`` is ~630 MB — check
the competition's upload limit. The probe also ships ``weights.joblib``.

If W&B is installed and logged in, each training run logs per-epoch training
loss/accuracy and the held-out benchmark metrics to ``neurips-eeg-2026-bci``
(override with ``WANDB_PROJECT``; ``WANDB_MODE=disabled`` turns it off). The
W&B run name matches the output folder, e.g. ``BENDR-full-shot5-seed42``.
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

BENDR_SFREQ = 256
HUB_ID = "braindecode/braindecode-bendr"
# Equivalent names for BENDR's 19 pretraining channels (10-20 vs 10-10).
ALIASES = {"T3": "T7", "T4": "T8", "P7": "T5", "P8": "T6"}
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


def _channel_projection(ch_names):
    """Fixed layer mapping the task's channels to BENDR's 19 EEG inputs.

    Channels BENDR was pretrained on are copied by name; the rest are
    spherical-spline interpolated from every task channel's standard 10-05
    position.
    """
    import mne
    from braindecode.models.bendr import _BENDR_TARGET_CHS_INFO
    from braindecode.modules import ChannelInterpolationLayer

    positions = {
        name.upper(): pos for name, pos in mne.channels.make_standard_montage(
            "standard_1005"
        ).get_positions()["ch_pos"].items()
    }
    src, unknown = [], []
    for name in ch_names or []:
        name = str(name).removeprefix("EEG ").split("-", 1)[0].upper()
        if name not in positions:
            unknown.append(name)
            continue
        src.append({"ch_name": ALIASES.get(name, name),
                    "loc": positions[name]})
    if unknown:
        raise ValueError("BENDR cannot place channels: " + ", ".join(unknown))
    target = _BENDR_TARGET_CHS_INFO[:19]
    names = {ch["ch_name"] for ch in src}
    missing = [ch["ch_name"] for ch in target if ch["ch_name"] not in names]
    print(f"[BENDR] {19 - len(missing)}/19 channels matched by name; "
          f"interpolated: {', '.join(missing) or 'none'}")
    return ChannelInterpolationLayer(src, target, mode="name_match")


def _build_backbone(n_times, pretrained):
    from braindecode.models import BENDR
    # No dropout: per-window feature differences are ~1% of the shared
    # start-token offset, smaller than dropout noise in train mode.
    kwargs = dict(n_outputs=1, n_times=n_times, sfreq=BENDR_SFREQ,
                  drop_prob=0.0)
    if pretrained:
        backbone = BENDR.from_pretrained(HUB_ID, **kwargs)
    else:
        backbone = BENDR(n_chans=20, **kwargs)
    backbone.final_layer = None
    return backbone


class BENDRClassifier(nn.Module):
    """BENDR (conv encoder + transformer) + standardized linear head on the
    contextualizer's start-token output, as in the BENDR paper.

    The start-token output is dominated by an offset shared by every window
    (per-window differences are ~1% of it), which a per-window LayerNorm
    cannot remove. The head therefore standardizes each feature across
    windows with BatchNorm (like the probe's StandardScaler); its running
    statistics follow the features as fine-tuning shifts that offset —
    fixed pre-training statistics blow up under even a tiny shift.
    """

    def __init__(self, meta, pretrained):
        super().__init__()
        self.sfreq = float(meta["sfreq"])
        self.n_times = max(
            round(meta["n_times"] * BENDR_SFREQ / self.sfreq), 1
        )
        self.project = _channel_projection(meta.get("ch_names"))
        # Peak-to-peak amplitude that maps to SCALE = +1; set from the
        # training windows in ``Solver.fit`` and saved with the weights.
        self.register_buffer("ref_range", torch.tensor(1.0))
        self.backbone = _build_backbone(self.n_times, pretrained)
        dim = self.backbone.encoder_h
        self.head = nn.Sequential(
            nn.BatchNorm1d(dim), nn.Linear(dim, meta["n_classes"])
        )

    def select(self, X):
        """``(B, C, T)`` task channels -> BENDR's 19 EEG channels."""
        X = torch.as_tensor(X, dtype=torch.float32)
        return self.project(X.to(self.ref_range.device))

    def features(self, X):
        X = self.select(X)
        if X.shape[-1] != self.n_times:
            X = F.interpolate(X, size=self.n_times, mode="linear",
                              align_corners=False)
        # DN3 / Deep1010 scaling: min-max each window to [-1, 1] and append
        # its relative amplitude as the SCALE channel.
        low = X.amin(dim=(1, 2), keepdim=True)
        high = X.amax(dim=(1, 2), keepdim=True)
        X = 2 * (X - low) / (high - low).clamp_min(1e-8) - 1
        scale = ((high - low) / self.ref_range).clamp(0, 1) * 2 - 1
        X = torch.cat([X, scale.expand(-1, 1, X.shape[-1])], dim=1)
        return self.backbone(X, return_features=True)["features"]  # (B, D)

    def forward(self, X):
        return self.head(self.features(X))


class _Probe(_LogsEvaluation):
    """Frozen BENDR features + scikit-learn ridge head."""

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

    name = "BENDR"
    micro_batch_size = 16

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
            "BENDR", self.strategy, f"shot{shot_label}"
        )
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        training = self.train_loader is not None
        if not training and not weights.is_file():
            raise FileNotFoundError(
                "BENDR inference needs weights.pt. Train with "
                "training=True and upload the exported submission folder."
            )
        pretrained = training and self.strategy != "scratch"
        model = BENDRClassifier(meta, pretrained=pretrained)
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
            ctx = backbone.contextualizer
            ctx.transformer_layers[-1].requires_grad_(True)
            ctx.norm.requires_grad_(True)
            ctx.output_layer.requires_grad_(True)
        elif self.strategy in ("full", "scratch"):
            backbone.requires_grad_(True)
        else:
            raise ValueError(f"Unknown BENDR strategy: {self.strategy}")
        model.head.requires_grad_(True)
        return _FineTuned(model)

    def fit(self, model, train_loader):
        signals, labels = self._collect(train_loader)
        if self.n_shot:
            signals, labels = self._sample_shots(signals, labels)
        bendr = model.model
        selected = bendr.select(signals).cpu()
        peak_to_peak = selected.amax(dim=(1, 2)) - selected.amin(dim=(1, 2))
        bendr.ref_range.fill_(peak_to_peak.max().item())
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
                "model": "BENDR",
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
                if len(indices) < 2:  # BatchNorm head needs >1 window
                    continue
                optimizer.zero_grad()
                # Gradient accumulation over micro-batches; BENDR fits a
                # full batch of 16 in ~3.2 GB, so this is one step.
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
