import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from torchmetrics.classification import BinaryAccuracy


class MultiHeadModel(pl.LightningModule):
    def __init__(self, n_regions, n_networks):
        super().__init__()

        self.n_regions = n_regions
        self.n_networks = n_networks

        self.train_accuracy = metrics.Accuracy(task='multiclass', num_classes=2)
        self.val_accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        # H3 (network-level timeseries)
        self.h1_encoder = nn.LSTM(n_regions, n_regions, batch_first=True)
        self.h1_decoder = nn.LSTM(n_regions, n_regions, batch_first=True)
        self.h1_output = nn.Sequential(
            nn.Linear(n_regions, 2),
            nn.Dropout(.2),
            nn.Sigmoid()
        )

        # TODO H2 (region-level connectivity)
        self.h2_head = nn.Linear(1, 1)

        # H3 (network-level timeseries)
        self.h3_encoder = nn.LSTM(n_networks, n_networks, batch_first=True)
        self.h3_decoder = nn.LSTM(n_networks, n_networks, batch_first=True)
        self.h3_output = nn.Sequential(
            nn.Linear(n_networks, 2),
            nn.Dropout(.2),
            nn.Sigmoid()
        )

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
        # H1
        batch_size, n_timepoints, _ = h1.shape
        _, z1 = self.h1_encoder(h1)
        x1_dec = torch.zeros(batch_size, n_timepoints, self.n_regions).to(self.device)
        h1_recon, _ = self.h1_decoder(x1_dec, z1)
        y1 = self.h1_output(z1[0].view(batch_size, -1))

        # H3
        # batch_size, n_timepoints, _ = h3.shape
        # _, z3 = self.h3_encoder(h3)
        # x3_dec = torch.zeros(batch_size, n_timepoints, self.n_networks).to(self.device)
        # h3_recon, _ = self.h3_decoder(x3_dec, z3)
        # y3 = self.h3_output(z3[0].view(batch_size, -1))

        # h4 = h4.view(h4.shape[0], -1)
        # y4 = self.h4_head(h4)

        # h5 = h5.view(h5.shape[0], -1)
        # y5 = self.h5_head(h5)

        # y = y4 + y5
        # y = self.h4_h5_output(y)

        return y1, h1_recon

    def training_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat, recon = self(h1, h2, h3, h4, h5)
        loss_cls = F.cross_entropy(y_hat, y)
        loss_recon = F.mse_loss(recon, h1)
        loss = loss_cls + loss_recon

        accuracy = self.train_accuracy(y_hat, y)

        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_recon', loss_recon)
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
        y_hat, recon = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.val_accuracy(y_hat, y.float())
        dropout_accuracy = self.calculate_dropout_accuracy(batch)

        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        self.log('val/dropout_accuracy', dropout_accuracy)

    def test_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat, recon = self(h1, h2, h3, h4, h5)
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
        for _ in range(n_repeats):
            y_hat, recon = self(h1, h2, h3, h4, h5)
            accuracy += self.val_accuracy(y_hat, y)
        accuracy /= n_repeats

        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
