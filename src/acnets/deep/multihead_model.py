import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl


class MultiHeadModel(pl.LightningModule):
    def __init__(self, n_timepoints, n_regions, n_networks):
        super().__init__()
        self.h1_head = nn.RNN(n_regions, 8, batch_first=True)
        self.h1_output = nn.Linear(8, 2)  # binary classification (AVGP or NVGP)
        self.h2_head = nn.Linear(1, 1)
        self.h3_head = nn.Linear(1, 1)
        self.h4_head = nn.Linear(1, 1)
        self.h5_head = nn.Linear(1, 1)

    def forward(self, h1, h2, h3, h4, h5):
        (z1, _) = self.h1_head(h1)
        y = self.h1_output(z1[:, -1, :])
        return y

    def training_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        h1, h2, h3, h4, h5, y = batch
        y_hat = self(h1, h2, h3, h4, h5)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
