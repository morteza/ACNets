from torch import nn
import pytorch_lightning as pl

class MultiScaleDeepModel(pl.LightningModule):
    def __init__(self, n_timepoints, n_regions, n_networks, n_subjects, n_outputs):
        super().__init__()
        self.h1_head = nn.RNN(n_timepoints, 1, batch_first=True)
        self.h2_head = nn.Linear(1, 1)
        self.h3_head = nn.Linear(1, 1)
        self.h4_head = nn.Linear(1, 1)
        self.h5_head = nn.Linear(1, 1)
        self.fc_output = nn.Linear(1, n_outputs)

    def forward(self, x):
        h1, h2, h3, h4, h5 = x
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)