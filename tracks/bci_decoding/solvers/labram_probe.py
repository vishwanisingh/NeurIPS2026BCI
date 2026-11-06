"""Frozen LaBraM-Base features with a trainable linear BCI classifier.

Training uses the official LaBraM source and checkpoint from
``~/.cache/labram/LaBraM`` (or ``LABRAM_ROOT``). Exported weights are a
self-contained TorchScript predictor, so inference does not need that source
tree or a network connection.
"""

import ast
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from benchmark_utils.base_solver import CompetSolver


def _input_channels(ch_names):
    root = Path(os.environ.get(
        "LABRAM_ROOT", Path.home() / ".cache/labram/LaBraM",
    ))
    utils_tree = ast.parse((root / "utils.py").read_text())
    standard_1020 = next(
        ast.literal_eval(node.value)
        for node in utils_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "standard_1020"
            for target in node.targets
        )
    )

    indices = [0]
    for channel in ch_names:
        name = str(channel).removeprefix("EEG ").split("-", 1)[0].upper()
        try:
            indices.append(standard_1020.index(name) + 1)
        except ValueError as err:
            raise ValueError(
                f"LaBraM has no spatial embedding for EEG channel {channel!r}"
            ) from err
    return indices


def _create_encoder(meta, pretrained=True):
    root = Path(os.environ.get(
        "LABRAM_ROOT", Path.home() / ".cache/labram/LaBraM",
    ))
    if not (root / "modeling_finetune.py").is_file():
        raise FileNotFoundError(
            f"LaBraM source not found at {root}; set LABRAM_ROOT to its folder"
        )
    sys.path.insert(0, str(root))
    import modeling_finetune

    encoder = modeling_finetune.labram_base_patch200_200(
        EEG_size=800,
        num_classes=0,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_values=0.1,
        qkv_bias=False,
        use_rel_pos_bias=False,
    )

    if pretrained:
        checkpoint_path = root / "checkpoints" / "labram-base.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"LaBraM checkpoint not found: {checkpoint_path}"
            )
        # This is the official checkpoint whose GitHub blob hash was verified.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        pretrained_state = {
            key.removeprefix("student."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("student.")
        }
        current = encoder.state_dict()
        compatible = {
            key: value for key, value in pretrained_state.items()
            if key in current and value.shape == current[key].shape
        }
        if len(compatible) < len(current) - 4:
            raise RuntimeError(
                "Too few LaBraM pretrained tensors matched the base encoder: "
                f"{len(compatible)} of {len(current)}"
            )
        encoder.load_state_dict(compatible, strict=False)
        encoder.requires_grad_(False)
    encoder.eval()
    return encoder


class LaBraMProbe(nn.Module):
    """LaBraM window classifier with a selectable trainable encoder scope."""

    def __init__(self, encoder, input_chans, n_chans, n_classes, device):
        super().__init__()
        self.encoder = encoder
        self.input_chans = input_chans
        self.n_chans = n_chans
        self.strategy = "probe"
        self.head = nn.Linear(200, n_classes)
        self.register_buffer("feature_mean", torch.zeros(1, 200))
        self.register_buffer("feature_std", torch.ones(1, 200))
        self.to(device)

    def encode(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        X = F.interpolate(X, size=800, mode="linear", align_corners=False)
        X = (X / 100.0).reshape(len(X), self.n_chans, 4, 200)
        return self.encoder.forward_features(
            X, input_chans=self.input_chans,
        )

    def forward(self, X):
        features = self.encode(X)
        features = (features - self.feature_mean) / self.feature_std
        return self.head(features)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)

    def log_evaluation(self, metrics):
        run_id = getattr(self, "wandb_run_id", None)
        if run_id is None or os.environ.get("WANDB_MODE", "online").lower() == "disabled":
            return
        try:
            import wandb
        except ImportError:
            return
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "neurips-eeg-2026-bci"),
            id=run_id,
            resume="allow",
            reinit=True,
        )
        run.log({f"evaluation/{key}": value for key, value in metrics.items()})
        run.finish()


class Solver(CompetSolver):

    name = "LaBraM"
    requirements = ["pip::timm==0.4.12"]
    parameters = {
        "strategy": ["probe", "last_block", "full", "scratch"],
        "n_shot": [0, 5, 10],
    }

    def set_objective(self, train_loader, **meta):
        self._original_name = self.name
        self._shot_label = "all" if self.n_shot == 0 else str(self.n_shot)
        self.name = f"LaBraM-{self.strategy}-shot{self._shot_label}"
        super().set_objective(train_loader, **meta)

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        if self.train_loader is None:
            if not weights.is_file():
                raise FileNotFoundError(
                    "Inference needs the exported LaBraM TorchScript weights"
                )
            return _TorchScriptPredictor(
                torch.jit.load(str(weights), map_location=self.device)
            )

        encoder = _create_encoder(meta, pretrained=self.strategy != "scratch")
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if self.strategy == "last_block":
            encoder.blocks[-1].requires_grad_(True)
            if encoder.fc_norm is not None:
                encoder.fc_norm.requires_grad_(True)
        elif self.strategy in ("full", "scratch"):
            encoder.requires_grad_(True)
        elif self.strategy != "probe":
            raise ValueError(f"Unknown LaBraM strategy: {self.strategy}")

        model = LaBraMProbe(
            encoder=encoder,
            input_chans=_input_channels(meta["ch_names"]),
            n_chans=meta["n_chans"],
            n_classes=meta["n_classes"],
            device=self.device,
        )
        model.strategy = self.strategy
        return model

    def fit(self, model, train_loader):
        run = self._start_wandb(model)
        try:
            X_train, y_train = self._training_examples(train_loader)
            loss_fn = nn.CrossEntropyLoss()
            if self.strategy == "probe":
                self._fit_probe(model, X_train, y_train, loss_fn, run)
            else:
                self._fine_tune(model, X_train, y_train, loss_fn, run)
        finally:
            if run is not None:
                model.wandb_run_id = run.id
                run.finish()

    def _training_examples(self, train_loader):
        signals, labels = [], []
        for X, y, _info in train_loader:
            signals.append(X.detach().cpu())
            labels.append(y.detach().reshape(-1).cpu().long())
        signals, labels = torch.cat(signals), torch.cat(labels)

        if self.n_shot == 0:
            return signals, labels

        self._selection_seed = torch.initial_seed()
        generator = torch.Generator().manual_seed(self._selection_seed)
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
        return signals[selected], labels[selected]

    def _fit_probe(self, model, X_train, y_train, loss_fn, run):
        features = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X_train), 64):
                features.append(model.encode(
                    X_train[start:start + 64].to(self.device)
                ))
        features = torch.cat(features)
        y_train = y_train.to(self.device)
        model.feature_mean.copy_(features.mean(dim=0, keepdim=True))
        model.feature_std.copy_(
            features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        )
        features = (features - model.feature_mean) / model.feature_std

        optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2)
        for epoch in range(100):
            optimizer.zero_grad()
            logits = model.head(features)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()
            self._log_epoch(run, epoch, loss, logits, y_train)

    def _fine_tune(self, model, X_train, y_train, loss_fn, run):
        model.encoder.eval()
        if self.strategy == "last_block":
            model.encoder.blocks[-1].train()
            if model.encoder.fc_norm is not None:
                model.encoder.fc_norm.train()
        else:
            model.encoder.train()

        encoder_params = [
            parameter for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": 5e-6},
                {"params": model.head.parameters(), "lr": 5e-4},
            ],
            weight_decay=0.05,
        )
        for epoch in range(20):
            order = torch.randperm(len(X_train))
            losses, correct, total = 0.0, 0, 0
            for indices in order.split(32):
                X_batch = X_train[indices].to(self.device)
                y_batch = y_train[indices].to(self.device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                losses += loss.detach().item() * len(indices)
                correct += (logits.detach().argmax(dim=1) == y_batch).sum().item()
                total += len(indices)
            epoch_loss = losses / max(total, 1)
            if run is not None:
                run.log({
                    "epoch": epoch,
                    "train/loss": epoch_loss,
                    "train/accuracy": correct / max(total, 1),
                })

    @staticmethod
    def _log_epoch(run, epoch, loss, logits, targets):
        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": loss.detach().item(),
                "train/accuracy": (
                    logits.detach().argmax(dim=1) == targets
                ).float().mean().item(),
            })

    def _start_wandb(self, model):
        try:
            import wandb
        except ImportError:
            return None

        if os.environ.get("WANDB_MODE", "online").lower() == "disabled":
            return None
        project = os.environ.get("WANDB_PROJECT", "neurips-eeg-2026-bci")
        shot_label = "all" if self.n_shot == 0 else str(self.n_shot)
        seed = torch.initial_seed()
        return wandb.init(
            project=project,
            name=f"{self.strategy}-shot{shot_label}-seed{seed}",
            config={
                "model": "LaBraM-Base",
                "strategy": self.strategy,
                "n_shot_per_class": self.n_shot or "all",
                "pretrained": self.strategy != "scratch",
                "n_classes": self.meta["n_classes"],
                "n_chans": self.meta["n_chans"],
                "n_times": self.meta["n_times"],
                "sfreq": self.meta["sfreq"],
                "seed": seed,
            },
            reinit=True,
        )

    @staticmethod
    def log_evaluation(model, metrics):
        run_id = getattr(model, "wandb_run_id", None)
        if run_id is None or os.environ.get("WANDB_MODE", "online").lower() == "disabled":
            return
        try:
            import wandb
        except ImportError:
            return
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "neurips-eeg-2026-bci"),
            id=run_id,
            resume="allow",
            reinit=True,
        )
        run.log({f"evaluation/{key}": value for key, value in metrics.items()})
        run.finish()

    def save_model(self, model, path):
        model.eval()
        example = torch.zeros(
            1, model.n_chans, 480, device=self.device,
        )
        predictor = torch.jit.trace(model, example, check_trace=False)
        predictor.save(str(path / "weights.pt"))


class _TorchScriptPredictor:

    def __init__(self, module):
        self.module = module

    @torch.inference_mode()
    def predict(self, X):
        return self.module(X).argmax(dim=1)