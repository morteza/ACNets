import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from torchmetrics.classification import BinaryAccuracy


class MultiHeadModel(pl.LightningModule):
    def __init__(self, n_regions, n_networks):
        super().__init__()

        self.train_accuracy = metrics.Accuracy(task='multiclass', num_classes=2)
        self.val_accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.h1_encoder = nn.LSTM(n_regions, 8, batch_first=True)
        self.h1_output = nn.Linear(8, 2)  # binary classification (AVGP or NVGP)
        self.h2_head = nn.Linear(1, 1)
        self.h3_encoder = nn.LSTM(n_networks, 8, batch_first=True)
        self.h3_output = nn.Linear(8, 2)
        self.h4_head = nn.Sequential(
            nn.Linear(n_networks * n_networks, n_networks),
        )
        self.h5_head = nn.Sequential(
            nn.Linear(n_networks * n_networks, n_networks),
        )

        self.h4_h5_output = nn.Sequential(
            nn.Linear(n_networks, n_networks),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(n_networks, 2),
            nn.Sigmoid()
        )

    def forward(self, h1, h2, h3, h4, h5):
        # (z1, _) = self.h1_head(h1)
        # y1 = self.h1_output(z1[:, -1, :])

        # (_, (z3, _)) = self.h3_encoder(h3)
        # y3 = self.h3_output(z3[0, ...])

        h4 = h4.view(h4.shape[0], -1)
        y4 = self.h4_head(h4)

        h5 = h5.view(h5.shape[0], -1)
        y5 = self.h5_head(h5)

        y = y4 + y5
        y = self.h4_h5_output(y)

        return y

    def training_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.train_accuracy(y_hat, y)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        return loss

    def enable_dropout(self):
        """Enables dropout during validation and test."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def validation_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.val_accuracy(y_hat, y.float())
        dropout_accuracy = self.calculate_dropout_accuracy(batch)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        self.log('val/dropout_accuracy', dropout_accuracy)

    def test_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.val_accuracy(y_hat, y)
        dropout_accuracy = self.calculate_dropout_accuracy(batch)
        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)
        self.log('test/dropout_accuracy', dropout_accuracy)

    def calculate_dropout_accuracy(self, batch, n_repeats=10):
        h1, h2, h3, h4, h5, y = batch
        self.enable_dropout()

        accuracy = 0
        for i in range(n_repeats):
            y_hat = self(h1, h2, h3, h4, h5)
            accuracy += self.val_accuracy(y_hat, y)
        accuracy /= n_repeats

        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
