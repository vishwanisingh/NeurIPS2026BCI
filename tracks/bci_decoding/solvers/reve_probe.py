"""Baseline — frozen REVE encoder + ridge-classifier linear probe.

A complete submission example: the frozen encoder is defined *here* (as
``REVEEncoder``) so you can swap it, the head, or the pooling freely. REVE
(braindecode, pretrained) maps each window to an embedding; a ridge-classifier
head on the mean-pooled embedding predicts the class. Only the head is
trained. Its weights travel as a joblib dump; REVE's pretrained weights come
from the worker's staged Hugging Face cache and are downloaded once on first
use in a local environment.

``LinearProbe`` / ``Encoder`` are shared infra in
``benchmark_utils/linear_probe.py`` — reused here to keep the example short;
inline them if you want to change the probe itself.
"""

import joblib
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.linear_model import RidgeClassifier

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.linear_probe import Encoder, LinearProbe


class REVEEncoder(Encoder):
    """Frozen pretrained braindecode REVE (200 Hz) -> ``(B, T', D)``.

    Needs a window of at least ~1 s (200 samples at 200 Hz); shorter windows
    raise in ``encode``.
    """

    REVE_SFREQ = 200
    PATCH_SIZE = 200

    def __init__(self, meta):
        from braindecode.models import REVE
        self.sfreq = float(meta["sfreq"])
        self.device = meta.get("device", "cpu")
        self._n_times = max(
            round(meta["n_times"] * self.REVE_SFREQ / self.sfreq), 1
        )
        if self._n_times < self.PATCH_SIZE:
            raise ValueError(
                "REVE needs at least one second of EEG after resampling "
                f"to 200 Hz; this window has {self._n_times} samples"
            )
        self.model = REVE.from_pretrained(
            "brain-bzh/reve-base",
            n_outputs=1,
            n_chans=meta["n_chans"],
            n_times=self._n_times,
            sfreq=self.REVE_SFREQ,
            chs_info=None,
        ).to(self.device)
        self._set_channel_positions(meta.get("ch_names"))
        self.model.eval()
        self.model.requires_grad_(False)

    def _set_channel_positions(self, ch_names):
        if not ch_names or len(ch_names) != self.model.n_chans:
            raise ValueError(
                "REVE needs one standard EEG channel name per input channel"
            )
        names = [str(name).removeprefix("EEG ").split("-", 1)[0]
                 for name in ch_names]
        known = self.model._position_bank.mapping
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                "REVE cannot locate these EEG channels: "
                + ", ".join(unknown)
            )
        self.model.default_pos = self.model.get_positions(names)

    def encode(self, X, grad=False):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if X.shape[-1] != self._n_times:
            X = F.interpolate(
                X, size=self._n_times, mode="linear", align_corners=False
            )
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X = ((X - mean) / std).clamp(-15.0, 15.0)
        if grad:
            out = self.model(X, return_features=True)
        else:
            with torch.inference_mode():
                out = self.model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        if feats.ndim == 4:      # (B, C, T', D) -> mean over channels
            feats = feats.mean(dim=1)
        elif feats.ndim == 2:    # (B, D) -> add a singleton time axis
            feats = feats[:, None, :]
        return feats             # (B, T', D)


class REVEFineTuner(nn.Module):
    """REVE encoder with a compact supervised BCI classification head."""

    def __init__(self, encoder, n_classes, device):
        super().__init__()
        self.encoder = encoder
        self.reve = encoder.model
        self.head = nn.Sequential(
            nn.LayerNorm(self.reve.embed_dim),
            nn.Linear(self.reve.embed_dim, n_classes),
        )
        self.to(device)

    def forward(self, X):
        features = self.encoder.encode(X, grad=True).mean(dim=1)
        return self.head(features)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)


class Solver(CompetSolver):

    name = "REVE"

    requirements = ["pip::braindecode", "pip::safetensors"]
    parameters = {"n_shot": [0, 5, 10]}
    parameters["strategy"] = ["probe", "last_block", "full"]

    def set_objective(self, train_loader, **meta):
        shot_label = "all" if self.n_shot == 0 else str(self.n_shot)
        self.name = self.run_label(
            "REVE", self.strategy, f"shot{shot_label}"
        )
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        encoder = REVEEncoder(meta)
        weights = meta["submission_dir"] / "weights.joblib"
        if self.strategy == "probe":
            probe = LinearProbe(encoder, RidgeClassifier())
            if weights.exists():
                print(f"[loading] {weights} into {self.name}")
                probe.head = joblib.load(weights)
            elif self.train_loader is None:
                raise FileNotFoundError(
                    "REVE probe inference needs weights.joblib. Train with "
                    "training=True and upload the exported submission folder."
                )
            return probe

        model = REVEFineTuner(encoder, meta["n_classes"], self.device)
        if self.strategy == "last_block":
            model.requires_grad_(False)
            model.reve.transformer.layers[-1].requires_grad_(True)
            model.head.requires_grad_(True)
        elif self.strategy == "full":
            model.requires_grad_(True)
        else:
            raise ValueError(f"Unknown REVE strategy: {self.strategy}")

        fine_tune_weights = meta["submission_dir"] / "weights.pt"
        if fine_tune_weights.exists() and self.train_loader is None:
            print(f"[loading] {fine_tune_weights} into {self.name}")
            model.load_state_dict(torch.load(
                fine_tune_weights, map_location=self.device, weights_only=True,
            ))
        elif self.train_loader is None:
            raise FileNotFoundError(
                "REVE fine-tuning inference needs weights.pt. Train with "
                "training=True and upload the exported submission folder."
            )
        return model

    def fit(self, model, train_loader):
        if self.n_shot == 0:
            if self.strategy == "probe":
                model.fit(train_loader)
            else:
                self._fit_finetune(model, self._collect(train_loader))
            return

        signals, labels = self._collect(train_loader)

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
        order = torch.randperm(len(selected), generator=generator)
        selected = selected[order]
        selected_data = [(signals[selected], labels[selected], {})]
        if self.strategy == "probe":
            model.fit(selected_data)
        else:
            self._fit_finetune(model, (signals[selected], labels[selected]))

    @staticmethod
    def _collect(train_loader):
        signals, labels = [], []
        for X, y, _info in train_loader:
            signals.append(X.detach().cpu())
            labels.append(y.detach().reshape(-1).cpu().long())
        return torch.cat(signals), torch.cat(labels)

    def _fit_finetune(self, model, training_data):
        signals, labels = training_data
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.reve.parameters() if p.requires_grad],
                    "lr": 5e-6,
                },
                {"params": model.head.parameters(), "lr": 5e-4},
            ],
            weight_decay=0.05,
        )
        loss_fn = nn.CrossEntropyLoss()
        for _epoch in range(20):
            model.train()
            order = torch.randperm(len(signals))
            for indices in order.split(16):
                X = signals[indices].to(self.device)
                y = labels[indices].to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                optimizer.step()

    def save_model(self, model, path):
        if self.strategy == "probe":
            joblib.dump(model.head, path / "weights.joblib")
        else:
            torch.save(model.state_dict(), path / "weights.pt")
