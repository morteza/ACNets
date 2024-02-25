from typing import Literal
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
import torchmetrics as metrics
from .cvae import CVAE

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


class WaveletModel(pl.LightningModule):
    def __init__(self, n_regions, n_wavelets=124, n_embeddings=2, segment_length=32):
        super().__init__()

        self.n_regions = n_regions
        self.n_embeddings = n_embeddings
        self.n_wavelets = n_wavelets
        self.segment_length = segment_length
        self.phase: Literal['pretrain', 'finetune', None] = None

        self.last_run_version = None

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.feature_extractor = CVAE(n_regions, n_embeddings, kernel_size=5, stride=2)
        self.cls_head = Classifier(n_embeddings)

    def forward(self, x_wvt):
        """_summary_

        Args:
            x_wvt (torch.Tensor): shape: (subjects, wavelets, regions)

        Returns:
            y, h, loss_recon
        """

        x = x_wvt[:, :self.n_wavelets, :]  # -> shape: (subjects, wavelets, regions)

        x_segments = x.unfold(1, self.segment_length, self.segment_length)
        x_segments = x_segments.reshape(-1, self.segment_length, self.n_regions)
        x_segments = x_segments.permute(0, 2, 1)  # -> shape: (subjects * segments, regions, segment_length)

        h, x_segments_recon, loss_recon = self.feature_extractor(x_segments)

        if self.phase == 'finetune':
            y = self.cls_head(h)
            y = y.reshape(x.size(0), -1, 2)
            y = y.mean(dim=1)
        else:
            y = None

        return y, h, loss_recon

    def set_phase(self, phase):
        self.phase = phase

    def step(self, batch, batch_idx, label: Literal['train', 'val', 'test'] = 'train'):
        x_wvt = batch[2]

        y_hat, h, loss_recon = self(x_wvt)
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

    def training_step(self, batch, batch_idx):
        if self.phase == 'finetune':
            self.unfreeze()
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

    def fit(self, datamodule, max_epochs=100,
            phase: Literal['pretrain', 'finetune', None] = None, **kwargs):

        self.set_phase(phase)

        match self.phase:
            case 'pretrain':
                run_name = 'wvt'
                ckpt_path = 'last'
                callbacks = [RichProgressBar(),
                             ModelCheckpoint(
                                 dirpath=f'models/checkpoints/{run_name}',
                                 filename='{epoch:02d}',
                                 every_n_epochs=10,
                                 every_n_train_steps=0,
                                 save_last=True)]
            case 'finetune':
                run_name = 'wvt_cls'
                ckpt_path = None
                callbacks = [RichProgressBar()]
            case _:
                raise ValueError(f'Invalid phase ({self.phase}). Must be "finetune" or "pretrain".')

        trainer = pl.Trainer(
            accelerator='auto',
            max_epochs=max_epochs,
            # accumulate_grad_batches=5,
            #  gradient_clip_val=.5,
            logger=TensorBoardLogger('lightning_logs', name=run_name,
                                     version=self.last_run_version),
            log_every_n_steps=1,
            callbacks=callbacks,
            **kwargs)

        self.last_run_version = 'version_{}'.format(trainer.logger.version) if trainer.logger else None

        trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt_path)

        return trainer
