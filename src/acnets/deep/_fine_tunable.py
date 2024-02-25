from typing import Literal
import pytorch_lightning as pl
import hashlib
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint


class FineTunable(pl.LightningModule):
    """Base class for fine-tunable models.

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

    def step(self, batch, batch_idx, label: Literal['train', 'val', 'test'] = 'train'):

        x = batch[0]
        y_hat, h, loss_recon = self(x)
        loss = loss_recon
        self.log(f'loss_recon/{label}', loss_recon)

        if self.phase == 'finetune':
            y = batch[-1]
            loss_cls = F.cross_entropy(y_hat, y)
            loss += loss_cls
            self.log(f'loss_cls/{label}', loss_cls)

            accuracy = self.accuracy(y_hat, y)
            self.accuracy.reset()
            self.log(f'accuracy/{label}', accuracy)

        return {'loss': loss}

    def on_train_start(self):
        # init custom hp metrics (default_hp_metric=False)
        self.logger.log_hyperparams(self.hparams, {
            'accuracy/val': torch.inf,
            'loss_recon/val': torch.inf,
            'loss_cls/val': torch.inf})

    def training_step(self, batch, batch_idx):
        self.unfreeze()
        if self.phase == 'pretrain':
            self.cls_head.freeze()
        return self.step(batch, batch_idx, label='train')

    def validation_step(self, batch, batch_idx):
        self.freeze()
        return self.step(batch, batch_idx, label='val')

    def test_step(self, batch, batch_idx):
        self.freeze()
        return self.step(batch, batch_idx, label='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def fit(self,
            datamodule,
            max_epochs=100,
            phase: Literal['pretrain', 'finetune', None] = None, **kwargs):

        callbacks: list = []  # [RichProgressBar()]
        run_name = (
            self.model_name
            if hasattr(self, 'model_name')
            else self.__class__.__name__)

        match phase:
            case 'pretrain':
                ckpt_path = 'last'
                ckpt_id = hashlib.md5(str(self.hparams).encode()).hexdigest()[:6]
                print(f'ckpt_id: {ckpt_id}')
                callbacks.append(ModelCheckpoint(
                    dirpath=f'checkpoints/{run_name}_{ckpt_id}',
                    filename=ckpt_id + '-{epoch}',
                    monitor='loss_recon/val',
                    every_n_epochs=1,
                    save_last=True))
            case 'finetune':
                run_name = f'{run_name}_ft'
                ckpt_path = None
            case _:
                raise ValueError(f'Invalid phase ({phase}). Must be "finetune" or "pretrain".')

        self.phase = phase

        trainer = pl.Trainer(
            accelerator='auto',
            max_epochs=max_epochs,
            accumulate_grad_batches=5,
             gradient_clip_val=.5,
            enable_progress_bar=False,
            logger=TensorBoardLogger('lightning_logs', name=run_name,
                                     default_hp_metric=False,
                                     version=self._last_run_version),
            log_every_n_steps=1,
            callbacks=callbacks,
            **kwargs)

        if phase == 'pretrain' and trainer.logger:
            self._last_run_version = 'version_{}'.format(trainer.logger.version)

        trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt_path)

        return trainer
