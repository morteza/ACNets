"""A small training loop, in place of pytorch_lightning.Trainer.

What Lightning was actually being used for here was modest: a loop over epochs
and batches, device placement, TensorBoard scalars, and a checkpoint at the end
of pretraining. That is what this module provides, in about the same number of
lines the Lightning subclasses took to configure.

The one thing worth stating explicitly, because Lightning did it invisibly, is
device selection. `resolve_device` defaults to CPU rather than to the fastest
available accelerator. On Apple Silicon, MPS silently computes several
operations in float32 regardless of the requested dtype and has historically
disagreed with CPU results on LSTM and ConvTranspose1d, both of which are used
here. The previous code carried a hack that reached into Keras global state to
force the device back to CPU after Keras had helpfully selected MPS; making CPU
the default and MPS opt-in is the same decision, taken once and in the open.

Models plug in by implementing `compute_loss`, which keeps them plain
`nn.Module`s that can be used, tested and pickled without a trainer anywhere in
sight:

    class MyModel(nn.Module):
        def compute_loss(self, batch, device):
            x = batch[0].to(device)
            ...
            return loss, {'loss_recon': float(loss_recon)}

Models needing two optimizers -- the GANs -- implement `training_step` instead
and drive their own backward passes; see `Trainer._run_batch`.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import nn

DeviceLike = str | torch.device | None


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Pick a device, defaulting to CPU.

    Set ACNETS_DEVICE=mps (or cuda) to opt in to an accelerator, or pass one
    explicitly. `resolve_device('auto')` picks the fastest available, which is
    what Lightning's `accelerator='auto'` did; it is not the default here.
    """
    if device is None:
        device = os.environ.get('ACNETS_DEVICE', 'cpu')
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy, replacing torchmetrics.Accuracy.

    torchmetrics was a stateful object that had to be `.reset()` after every
    read, and forgetting to was a live bug in the code this replaces. A
    function cannot accumulate state it forgot to clear.
    """
    predicted = (logits > 0.5).long() if logits.ndim == 1 else logits.argmax(dim=-1)
    targets = targets.argmax(dim=-1) if targets.ndim > 1 else targets.long()
    return float((predicted == targets).float().mean())


class MetricLogger:
    """TensorBoard scalars, or nothing at all.

    Writes to `runs/<name>/<version>` and mirrors the last value of each scalar
    in `history`, so tests and notebooks can assert on training behaviour
    without parsing event files.
    """

    def __init__(self,
                 name: str,
                 root: str | Path = 'runs',
                 version: str | None = None,
                 enabled: bool = True):
        self.name = name
        self.history: dict[str, list[float]] = {}
        self._writer = None

        if not enabled:
            self.log_dir = None
            self.version = version
            return

        root = Path(root) / name
        if version is None:
            existing = sorted(root.glob('version_*')) if root.exists() else []
            version = f'version_{len(existing)}'
        self.version = version
        self.log_dir = root / version

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(float(value))
            if self._writer is not None:
                self._writer.add_scalar(key, float(value), step)

    def log_hyperparams(self, hparams: Mapping) -> None:
        if self._writer is None:
            return
        clean = {k: v for k, v in hparams.items()
                 if isinstance(v, (int, float, str, bool, torch.Tensor))}
        self._writer.add_hparams(clean, {})

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def last(self, key: str) -> float | None:
        values = self.history.get(key)
        return values[-1] if values else None


@dataclass
class Trainer:
    """Epoch loop, device placement, logging and checkpointing.

    Parameters
    ----------
    max_epochs
        Epochs to run.
    device
        See `resolve_device`. CPU unless asked otherwise.
    checkpoint_dir
        When set, the model state dict is written here after every epoch as
        `last.pt`. This is the `save_last=True` behaviour the pretraining phase
        relied on to hand weights to the fine-tuning phase.
    logger
        A MetricLogger, or None for a silent run. Tests use `Trainer(...,
        logger=MetricLogger('x', enabled=False))` to keep the history without
        touching the filesystem.
    """

    max_epochs: int = 100
    device: DeviceLike = None
    lr: float = 1e-3
    checkpoint_dir: str | Path | None = None
    logger: MetricLogger | None = None
    grad_clip: float | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.device = resolve_device(self.device)

    def _optimizers(self, model: nn.Module) -> list[torch.optim.Optimizer]:
        if hasattr(model, 'configure_optimizers'):
            optimizers = model.configure_optimizers()
            if isinstance(optimizers, torch.optim.Optimizer):
                return [optimizers]
            return list(optimizers)
        return [torch.optim.Adam(model.parameters(), lr=self.lr)]

    def _run_batch(self, model, batch, optimizers) -> dict[str, float]:
        """One optimisation step.

        Models with a single loss implement `compute_loss` and the trainer owns
        the backward pass. Models that alternate between two optimizers -- a
        generator and a discriminator -- implement `training_step(batch,
        optimizers, device)` and own it themselves, because interleaving two
        backward passes correctly is model-specific and not something a generic
        loop should guess at.
        """
        if hasattr(model, 'training_step'):
            return model.training_step(batch, optimizers, self.device)

        loss, metrics = model.compute_loss(batch, self.device)
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        for optimizer in optimizers:
            optimizer.step()
        # detach before float(): the loss still carries a graph at this point,
        # and torch warns that reading a scalar off it is rarely what you meant
        return {'loss': float(loss.detach()), **metrics}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: Iterable,
                 label: str = 'val') -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            if hasattr(model, 'validation_step'):
                metrics = model.validation_step(batch, self.device)
            else:
                loss, metrics = model.compute_loss(batch, self.device)
                metrics = {'loss': float(loss), **metrics}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
        model.train()
        if n_batches == 0:
            return {}
        return {f'{key}/{label}': value / n_batches for key, value in totals.items()}

    def fit(self,
            model: nn.Module,
            train_loader: Iterable,
            val_loader: Iterable | None = None) -> nn.Module:
        model = model.to(self.device)
        model.train()
        optimizers = self._optimizers(model)

        if self.logger is not None and hasattr(model, 'hparams'):
            self.logger.log_hyperparams(model.hparams)

        for epoch in range(self.max_epochs):
            epoch_totals: dict[str, float] = {}
            n_batches = 0

            for batch in train_loader:
                metrics = self._run_batch(model, batch, optimizers)
                for key, value in metrics.items():
                    epoch_totals[key] = epoch_totals.get(key, 0.0) + value
                n_batches += 1

            record = {f'{k}/train': v / max(1, n_batches)
                      for k, v in epoch_totals.items()}
            if val_loader is not None:
                record.update(self.evaluate(model, val_loader))

            record['epoch'] = epoch
            self.history.append(record)
            if self.logger is not None:
                self.logger.log({k: v for k, v in record.items() if k != 'epoch'},
                                step=epoch)

            if self.checkpoint_dir is not None:
                path = Path(self.checkpoint_dir)
                path.mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                           path / 'last.pt')

        if self.logger is not None:
            self.logger.close()
        return model


def checkpoint_id(hparams: Mapping) -> str:
    """Stable six-character id for a hyperparameter set.

    Carried over from the Lightning code so existing `checkpoints/` directories
    keep their names.
    """
    return hashlib.md5(str(dict(sorted(hparams.items()))).encode()).hexdigest()[:6]


def load_checkpoint(model: nn.Module, path: str | Path,
                    strict: bool = True) -> nn.Module:
    """Restore weights saved by `Trainer.fit`."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload)
    model.load_state_dict(state, strict=strict)
    return model


Phase = Literal['pretrain', 'finetune']
