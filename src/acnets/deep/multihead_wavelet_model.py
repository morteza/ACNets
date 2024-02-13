from typing import Literal
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from .seq2seq import Seq2SeqAutoEncoder
from .vae import VariationalAutoEncoder
from .cvae import CVAE


class MultiHeadWaveletModel(pl.LightningModule):
    def __init__(self, n_regions, n_wavelets=32, n_embeddings=2):
        super().__init__()

        self.n_regions = n_regions
        self.n_embeddings = n_embeddings
        self.n_wavelets = n_wavelets
        self.enable_cls_head = False

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.cvae_head = CVAE(n_regions, n_embeddings, kernel_size=5, stride=2)

        self.cls_head = nn.Sequential(
            nn.Linear(n_embeddings, 2),
            nn.Sigmoid()
        )

    def forward(self, x_wvt):

        x = x_wvt[:, :self.n_wavelets, :].transpose(1, 2)  # -> shape: (subjects, regions, wavelets)
        h, x_recon, loss = self.cvae_head(x)

        if self.enable_cls_head:
            y = self.cls_head(h)
            return h, loss, y

        return h, loss, None

    def step(self, batch, batch_idx, phase: Literal['train', 'test', 'val'] = 'train'):
        x_wvt = batch[5]

        h, loss_recon, y_hat = self(x_wvt)
        loss = loss_recon
        self.log(f'{phase}/loss_recon', loss_recon)

        if self.enable_cls_head:
            y = batch[6]
            loss_cls = F.cross_entropy(y_hat, y)
            self.log(f'{phase}/loss_cls', loss_cls)

            loss += loss_cls

            accuracy = self.accuracy(y_hat, y)
            self.accuracy.reset()
            self.log(f'{phase}/accuracy', accuracy)

        self.log(f'{phase}/loss', loss)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        self.unfreeze()
        return self.step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        self.freeze()
        return self.step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        self.freeze()
        return self.step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
