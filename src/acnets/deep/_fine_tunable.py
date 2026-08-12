"""Two-phase models: reconstruct first, then classify.

`FineTunable` is a plain `nn.Module` that knows which of two phases it is in.
In `pretrain` only the reconstruction loss is optimised and the classifier head
is frozen; in `finetune` the classification loss is added and everything trains.

Subclasses implement `forward(x) -> (y_hat, h, loss_recon)` and own a
`cls_head`. Everything else -- the loop, the device, the logging, the
checkpoint -- lives in `acnets.deep.training`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .training import (
    MetricLogger,
    Trainer,
    accuracy,
    checkpoint_id,
    load_checkpoint,
)


class FineTunable(nn.Module):
    """Base class for models pretrained on reconstruction, then fine-tuned.

    Raises:
        ValueError: when phase is not `finetune` or `pretrain`.
    """

    _phase: Literal['pretrain', 'finetune', None] = None
    _last_run_version: str | None = None  # keep track of the last pretrained model version

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: Literal['pretrain', 'finetune']):
        if phase not in ['pretrain', 'finetune']:
            raise ValueError(f'Invalid phase ({phase}). '
                             'Must be `finetune` or `pretrain`.')
        self._phase = phase

    @property
    def hparams(self) -> dict:
        return getattr(self, '_hparams', {})

    def save_hyperparameters(self, **kwargs) -> None:
        """Record constructor arguments, as Lightning's method of the same name did.

        Explicit rather than magical: Lightning inspected the caller's frame to
        find them, which worked until a subclass added a `*args`. Pass what you
        want kept.
        """
        self._hparams = dict(kwargs)

    def compute_loss(self, batch, device) -> tuple[torch.Tensor, dict]:
        """Reconstruction loss, plus classification loss when fine-tuning."""
        x = batch[0].to(device)
        y_hat, _h, loss_recon = self(x)

        loss = loss_recon
        metrics = {'loss_recon': float(loss_recon.detach())}

        if self.phase == 'finetune':
            y = batch[-1].to(device)
            loss_cls = F.cross_entropy(y_hat, y)
            loss = loss + loss_cls
            metrics['loss_cls'] = float(loss_cls.detach())
            metrics['accuracy'] = accuracy(y_hat.detach(), y)

        return loss, metrics

    @torch.no_grad()
    def validation_step(self, batch, device) -> dict:
        loss, metrics = self.compute_loss(batch, device)
        return {'loss': float(loss), **metrics}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def freeze_classifier(self, frozen: bool = True) -> None:
        """Freeze the classification head, as the pretraining phase requires."""
        if hasattr(self, 'cls_head'):
            for parameter in self.cls_head.parameters():
                parameter.requires_grad_(not frozen)

    def fit(self,
            datamodule,
            max_epochs: int = 100,
            phase: Literal['pretrain', 'finetune', None] = None,
            device=None,
            log_root: str | Path = 'runs',
            **kwargs) -> Trainer:
        """Train for one phase.

        `pretrain` writes `checkpoints/<run_name>_<id>/last.pt` and freezes the
        classifier head; `finetune` logs under a separate run name and trains
        everything. Returns the Trainer, whose `history` holds the per-epoch
        metrics.
        """
        run_name = getattr(self, 'model_name', self.__class__.__name__)

        match phase:
            case 'pretrain':
                run_id = checkpoint_id(self.hparams)
                checkpoint_dir = Path('checkpoints') / f'{run_name}_{run_id}'
                self.freeze_classifier(True)
            case 'finetune':
                run_name = f'{run_name}_ft'
                checkpoint_dir = None
                self.freeze_classifier(False)
            case _:
                raise ValueError(
                    f'Invalid phase ({phase}). Must be "finetune" or "pretrain".')

        self.phase = phase

        logger = MetricLogger(run_name, root=log_root,
                              version=self._last_run_version)
        trainer = Trainer(max_epochs=max_epochs,
                          device=device,
                          checkpoint_dir=checkpoint_dir,
                          logger=logger,
                          **kwargs)

        if phase == 'pretrain':
            self._last_run_version = logger.version

        datamodule.setup('fit')
        trainer.fit(self, datamodule.train_dataloader(),
                    datamodule.val_dataloader())
        return trainer

    def load_pretrained(self, path: str | Path, strict: bool = False):
        """Load weights written by the pretraining phase."""
        return load_checkpoint(self, path, strict=strict)
