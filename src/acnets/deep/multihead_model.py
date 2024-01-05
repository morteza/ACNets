import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from .seq2seq import Seq2SeqAutoEncoder


class MultiHeadModel(pl.LightningModule):
    def __init__(self, n_regions, n_networks, n_embeddings=32):
        super().__init__()

        self.n_regions = n_regions
        self.n_networks = n_networks
        self.n_embeddings = n_embeddings

        self.train_accuracy = metrics.Accuracy(task='multiclass', num_classes=2)
        self.val_accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        # X1 (region-level timeseries)
        self.x1_autoencoder = Seq2SeqAutoEncoder(n_regions, n_embeddings)
        self.x1_head = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings)
        )

        # X2 (region-level connectivity)
        self.x2_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_regions * n_regions, n_embeddings)
        )

        # X3 (network-level timeseries)
        self.x3_autoencoder = Seq2SeqAutoEncoder(n_networks, n_embeddings)
        self.x3_head = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings)
        )

        # X4 (network-level connectivity)
        self.x4_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_networks * n_networks, n_embeddings)
        )

        # X5 (averaged network-level connectivity)
        self.x5_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_networks * n_networks, n_embeddings)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(n_embeddings * 5, n_embeddings),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(n_embeddings, 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4, x5):

        h1, x1_recon = self.x1_autoencoder(x1)
        h1 = self.x1_head(h1)

        h2 = self.x2_head(x2)

        h3, x3_recon = self.x3_autoencoder(x3)
        h3 = self.x3_head(h3)

        h4 = self.x4_head(x4)

        h5 = self.x5_head(x5)

        h = torch.cat([h1, h2, h3, h4, h5], dim=1)
        y = self.cls_head(h)

        return y, x1_recon, x3_recon

    def training_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, y = batch
        y_hat, x1_recon, x3_recon = self(x1, x2, x3, x4, x5)
        loss_cls = F.cross_entropy(y_hat, y)
        loss_recon_x1 = F.mse_loss(x1_recon, x1) / 100000
        loss_recon_x3 = F.mse_loss(x3_recon, x3) / 100000
        loss = loss_cls + loss_recon_x1 + loss_recon_x3

        accuracy = self.train_accuracy(y_hat, y)

        self.log('train/loss_cls', loss_cls)
        # self.log('train/loss_recon', loss_recon)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        return loss

    def enable_dropout(self):
        """Enables dropout during validation and test."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, y = batch
        y_hat, *_ = self(x1, x2, x3, x4, x5)
        loss_cls = F.cross_entropy(y_hat, y)

        accuracy = self.val_accuracy(y_hat, y.float())
        dropout_accuracy = self.calculate_dropout_accuracy(batch)

        self.log('val/loss_cls', loss_cls)
        self.log('val/accuracy', accuracy)
        self.log('val/dropout_accuracy', dropout_accuracy)

    def test_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, y = batch
        y_hat, *_ = self(x1, x2, x3, x4, x5)
        loss_cls = F.cross_entropy(y_hat, y)

        accuracy = self.val_accuracy(y_hat, y)
        dropout_accuracy = self.calculate_dropout_accuracy(batch)

        self.log('test/loss_cls', loss_cls)
        self.log('test/accuracy', accuracy)
        self.log('test/dropout_accuracy', dropout_accuracy)

    def calculate_dropout_accuracy(self, batch, n_repeats=10):
        x1, x2, x3, x4, x5, y = batch
        self.enable_dropout()

        accuracy = 0
        for _ in range(n_repeats):
            y_hat, *_ = self(x1, x2, x3, x4, x5)
            accuracy += self.val_accuracy(y_hat, y)
        accuracy /= n_repeats

        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
