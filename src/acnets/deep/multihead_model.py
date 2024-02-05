import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from .seq2seq import Seq2SeqAutoEncoder
from .vae import VariationalAutoEncoder


class MultiHeadModel(pl.LightningModule):
    def __init__(self, n_regions, n_networks, n_embeddings=2):
        super().__init__()

        self.n_regions = n_regions
        self.n_networks = n_networks
        self.n_embeddings = n_embeddings

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        # X1 (region-level timeseries)
        # self.x1_autoencoder = Seq2SeqAutoEncoder(n_regions, n_embeddings)
        # self.x1_head = nn.Sequential(
        #     nn.Linear(n_embeddings, n_embeddings)
        # )

        # X2 (region-level connectivity)
        # self.x2_head = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     VariationalAutoEncoder(n_regions * n_regions, n_embeddings)
        # )

        # X3 (network-level timeseries)
        # self.x3_autoencoder = Seq2SeqAutoEncoder(n_networks, n_embeddings)
        # self.x3_head = nn.Sequential(
        #     nn.Linear(n_embeddings, n_embeddings)
        # )

        # X4 (network-level connectivity)
        self.x4_head = nn.Sequential(
            # nn.Flatten(start_dim=1),
            VariationalAutoEncoder(n_networks * (n_networks - 1) // 2, n_embeddings)
        )

        # X5 (averaged network-level connectivity)
        self.x5_head = nn.Sequential(
            # nn.Flatten(start_dim=1),
            VariationalAutoEncoder(n_networks * (n_networks + 1) // 2, n_embeddings)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings // 2),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(n_embeddings // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4, x5):

        # h1, x1_recon = self.x1_autoencoder(x1)
        # h1 = self.x1_head(h1)

        # h2, x2_recon = self.x2_head(x2)
        # x2_recon = x2_recon.view(-1, self.n_regions, self.n_regions)

        # h3, x3_recon = self.x3_autoencoder(x3)
        # h3 = self.x3_head(h3)

        h4, x4_recon, loss_x4 = self.x4_head(x4)
        h5, x5_recon, loss_x5 = self.x5_head(x5)

        loss_recon = loss_x4 + loss_x5

        h = torch.stack([h4, h5], dim=0).sum(dim=0)
        y = self.cls_head(h)

        return y, loss_recon

    def fine_tune(self, x1, x2, x3, x4, x5, y):
        raise NotImplementedError('Fine-tuning is not implemented yet.')

    def training_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, y = batch

        # keep only the triu of the x4 and flatten non-zero elements
        x4 = x4[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=1) == 1]
        x5 = x5[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=0) == 1]

        y_hat, loss_recon = self(x1, x2, x3, x4, x5)

        beta = 1.  # reconstruction loss coefficient
        loss_recon = beta * loss_recon
        loss_cls = F.cross_entropy(y_hat, y)
        loss = loss_cls + loss_recon

        accuracy = self.accuracy(y_hat, y)
        self.accuracy.reset()

        self.log('train/accuracy', accuracy)
        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss', loss)

        return {'loss': loss, 'train/accuracy': accuracy}

    def enable_dropout(self):
        """Enables dropout during validation and test."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def validation_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, y = batch

        # keep only the triu of the x4 and flatten non-zero elements
        x4 = x4[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=1) == 1]
        x5 = x5[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=0) == 1]

        y_hat, *_ = self(x1, x2, x3, x4, x5)

        loss_cls = F.cross_entropy(y_hat, y)

        accuracy = self.accuracy(y_hat, y.float())
        self.accuracy.reset()
        dropout_accuracy = self.calculate_dropout_accuracy(batch)

        self.log('val/loss_cls', loss_cls, sync_dist=True)
        self.log('val/accuracy', accuracy, sync_dist=True)
        self.log('val/dropout_accuracy', dropout_accuracy, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x1, x2, x3, x4, x5, y = batch

        # keep only the triu of the x4 and flatten non-zero elements
        x4 = x4[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=1) == 1]
        x5 = x5[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=0) == 1]

        y_hat, *_ = self(x1, x2, x3, x4, x5)

        loss_cls = F.cross_entropy(y_hat, y)

        accuracy = self.accuracy(y_hat, y)
        self.accuracy.reset()
        dropout_accuracy = self.calculate_dropout_accuracy(batch)

        self.log('test/loss_cls', loss_cls, sync_dist=True)
        self.log('test/accuracy', accuracy, sync_dist=True)
        self.log('test/dropout_accuracy', dropout_accuracy, sync_dist=True)

    def calculate_dropout_accuracy(self, batch, n_repeats=10):
        x1, x2, x3, x4, x5, y = batch

        x4 = x4[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=1) == 1]
        x5 = x5[:, torch.triu(torch.ones(self.n_networks, self.n_networks), diagonal=0) == 1]

        self.enable_dropout()

        accuracy = 0
        for _ in range(n_repeats):
            y_hat, *_ = self(x1, x2, x3, x4, x5)
            accuracy += self.accuracy(y_hat, y)
            self.accuracy.reset()
        accuracy /= n_repeats

        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
