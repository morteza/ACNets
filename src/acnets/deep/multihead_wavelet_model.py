from typing import Literal
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
import torchmetrics as metrics
from .seq2seq import Seq2SeqAutoEncoder
from .vae import VariationalAutoEncoder
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


class MultiHeadWaveletModel(pl.LightningModule):
    def __init__(self, n_regions, n_wavelets=124, n_embeddings=2, segment_length=32):
        super().__init__()

        self.n_regions = n_regions
        self.n_embeddings = n_embeddings
        self.n_wavelets = n_wavelets
        self.segment_length = segment_length

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

        if self._is_finetune_enabled:
            y = self.cls_head(h)
            y = y.reshape(x.size(0), -1, 2)
            y = y.mean(dim=1)
        else:
            y = None

        return y, h, loss_recon

    def step(self, batch, batch_idx, phase: Literal['train', 'test', 'val'] = 'train'):
        x_wvt = batch[5]

        y_hat, h, loss_recon = self(x_wvt)
        loss = loss_recon
        self.log(f'loss_recon/{phase}', loss_recon)

        if y_hat is not None:
            y = batch[6]
            loss_cls = F.cross_entropy(y_hat, y)
            loss += loss_cls
            self.log(f'loss_cls/{phase}', loss_cls)

            accuracy = self.accuracy(y_hat, y)
            self.accuracy.reset()
            self.log(f'accuracy/{phase}', accuracy)

        return {'loss': loss}

    def enable_finetune(self):
        self.unfreeze()
        # self.feature_extractor.decoder.freeze()
        self.feature_extractor.freeze()
        self._is_finetune_enabled = True

    def disable_finetune(self):
        self.unfreeze()
        self.cls_head.freeze()
        self._is_finetune_enabled = False

    def training_step(self, batch, batch_idx):
        if self._is_finetune_enabled:
            self.enable_finetune()
            return self.step(batch, batch_idx, phase='train')
        else:
            self.disable_finetune()
            return self.step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        self.freeze()
        return self.step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        self.freeze()
        return self.step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def fit(self, datamodule, max_epochs=100, with_classifier=False, **kwargs):
        # pre-train
        if with_classifier:
            run_name = 'wvt_cls'
            ckpt_path = None
            callbacks = [RichProgressBar()]
            self.enable_finetune()
        else:
            run_name = 'wvt'
            ckpt_path = 'last'
            callbacks = [RichProgressBar(),
                         ModelCheckpoint(
                             dirpath='models/checkpoints/',
                             filename='wvt-{epoch:02d}',
                             every_n_epochs=10,
                             every_n_train_steps=0,
                             save_last=True)]
            self.disable_finetune()

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

        self.last_run_version = 'version_{}'.format(trainer.logger.version)

        trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt_path)

        return trainer
