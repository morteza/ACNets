import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Encoder(pl.LightningModule):
    def __init__(self, n_inputs, n_ouputs):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(n_inputs, n_inputs * 2),
            nn.ReLU(),
            nn.Linear(n_inputs * 2, n_inputs * 4),
            nn.ReLU(),
            nn.Linear(n_inputs * 4, n_ouputs),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(pl.LightningModule):
    def __init__(self, n_inputs, n_ouputs):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(n_inputs, n_inputs // 2),
            nn.ReLU(),
            nn.Linear(n_inputs // 2, n_inputs // 4),
            nn.ReLU(),
            nn.Linear(n_inputs // 4, n_ouputs),
        )

    def forward(self, x):
        out = self.decode(x)
        return out


class CausalVAE(pl.LightningModule):
    def __init__(self, n_inputs, n_embeddings, mask_size):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.mask_size = mask_size
        self.encoder = Encoder(n_inputs, n_embeddings)
        self.decoder = Decoder(n_embeddings, n_inputs)

        self.mean_fc = nn.Linear(n_embeddings, n_embeddings)
        self.logvar_fc = nn.Linear(n_embeddings, n_embeddings)

    def reparameterize(self, mean, logvar):
        # re-parameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):

        # mask n items from the end of x
        x_masked = x.clone()
        x_masked[..., -self.mask_size:] = 0.0

        h = self.encoder(x_masked)

        mean = self.mean_fc(h)
        logvar = self.logvar_fc(h)

        # re-parameterization trick
        z = self.reparameterize(mean, logvar)

        x_recon = self.decoder(z)

        loss = self.loss(x[..., -self.mask_size:],
                         x_recon[..., -self.mask_size:],
                         mean, logvar)

        return h, x_recon, loss

    def loss(self, x, x_recon, mean, logvar):
        # reconstruction loss
        recon_loss = F.mse_loss(x, x_recon)
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return recon_loss + kl_loss
