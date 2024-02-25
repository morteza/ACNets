from typing import Literal
import hashlib
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
import torchmetrics as metrics
from .mvae import MaskedVAE
from .seq2seq import Seq2SeqAE


class Classifier(pl.LightningModule):
    def __init__(self, n_inputs=32):
        super().__init__()

        self.n_inputs = n_inputs

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.model = nn.Sequential(
            nn.Linear(n_inputs, n_inputs),
            nn.ReLU(),
            nn.Linear(n_inputs, n_inputs // 2),
            nn.ReLU(),
            nn.Linear(n_inputs // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MaskedModel(pl.LightningModule):
    def __init__(self, n_regions, n_embeddings=2):
        super().__init__()

        self.save_hyperparameters()

        self.phase: Literal['pretrain', 'finetune', None] = None

        self.last_run_version = None

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        # self.feature_extractor = MaskedVAE(segment_length, n_embeddings, mask_size=4)
        self.feature_extractor = Seq2SeqAE(n_regions, n_embeddings, mask_size=4)
        self.cls_head = Classifier(n_inputs=n_embeddings)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): shape: (subjects, timepoints, regions)

        Returns:
            y, h, loss_recon
        """

        h, x_recon, loss_recon = self.feature_extractor(x)

        if self.phase == 'finetune':
            y = self.cls_head(h)
        else:
            y = None

        return y, h, loss_recon

    def set_phase(self, phase):
        self.phase = phase

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
        if self.phase == 'finetune':
            self.unfreeze()
            # self.cls_head.unfreeze()
            # self.feature_extractor.decoder.freeze()
            # self.feature_extractor.freeze()
            return self.step(batch, batch_idx, label='train')
        elif self.phase == 'pretrain':
            self.unfreeze()
            self.cls_head.freeze()
            return self.step(batch, batch_idx, label='train')
        else:
            raise ValueError(f'Invalid phase ({self.phase}). Must be "finetune" or "pretrain".')

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

        self.set_phase(phase)

        callbacks: list = []  # [RichProgressBar()]
        run_name: str = 'S2SAE'

        match self.phase:
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
                run_name += '_ft'
                ckpt_path = None
            case _:
                raise ValueError(f'Invalid phase ({self.phase}). Must be "finetune" or "pretrain".')

        trainer = pl.Trainer(
            accelerator='auto',
            max_epochs=max_epochs,
            # accumulate_grad_batches=5,
            #  gradient_clip_val=.5,
            enable_progress_bar=False,
            logger=TensorBoardLogger('lightning_logs', name=run_name,
                                     default_hp_metric=False,
                                     version=self.last_run_version),
            log_every_n_steps=1,
            callbacks=callbacks,
            **kwargs)

        self.last_run_version = 'version_{}'.format(trainer.logger.version) if trainer.logger else None

        trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt_path)

        return trainer
