from typing import Literal
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
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
    def __init__(self, n_regions, n_wavelets=32, n_embeddings=2):
        super().__init__()

        self.n_regions = n_regions
        self.n_embeddings = n_embeddings
        self.n_wavelets = n_wavelets

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.feature_extractor = CVAE(n_regions, n_embeddings, kernel_size=5, stride=2)
        self.cls_head = Classifier(n_embeddings)

    def forward(self, x_wvt):

        x = x_wvt[:, :self.n_wavelets, :].transpose(1, 2)  # -> shape: (subjects, regions, wavelets)
        h, x_recon, loss_recon = self.feature_extractor(x)

        y = self.cls_head(h) if self._is_finetune_enabled else None

        return h, loss_recon, y

    def step(self, batch, batch_idx, phase: Literal['train', 'test', 'val'] = 'train'):
        x_wvt = batch[5]

        h, loss_recon, y_hat = self(x_wvt)
        loss = loss_recon
        self.log(f'loss_recon/{phase}', loss_recon)

        if y_hat is not None:
            y = batch[6]
            loss_cls = F.cross_entropy(y_hat, y)
            loss = loss_cls
            self.log(f'loss_cls/{phase}', loss_cls)

            accuracy = self.accuracy(y_hat, y)
            self.accuracy.reset()
            self.log(f'accuracy/{phase}', accuracy)

        return {'loss': loss}

    def enable_finetune(self):
        self.unfreeze()
        self.feature_extractor.decoder.freeze()
        self._is_finetune_enabled = True

    def disable_finetune(self):
        self.unfreeze()
        self.cls_head.freeze()
        self._is_finetune_enabled = False

    def training_step(self, batch, batch_idx):
        if self._is_finetune_enabled:
            self.enable_finetune()
            return self.step(batch, batch_idx, phase='finetune')
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
