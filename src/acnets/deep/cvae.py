import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_channels, n_embeddings):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_channels * 8, n_embeddings)
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self, n_channels, n_embeddings):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(n_embeddings, n_channels * 8),
            nn.Unflatten(dim=1, unflattened_size=(n_channels * 8, 1)),
            nn.ConvTranspose1d(n_channels * 8, n_channels * 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels * 4, n_channels * 2, kernel_size=5, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels * 2, n_channels, kernel_size=5, stride=2, output_padding=1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.decode(x)
        return out


class CVAE(nn.Module):
    def __init__(self, n_channels, n_embeddings, kernel_size=5, stride=2):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.encode = Encoder(n_channels, n_embeddings)
        self.decode = Decoder(n_channels, n_embeddings)

        self.mean_fc = nn.Linear(n_embeddings, n_embeddings)
        self.logvar_fc = nn.Linear(n_embeddings, n_embeddings)

    def reparameterize(self, mean, logvar):
        # re-parameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):

        h = self.encode(x)

        mean = self.mean_fc(h)
        logvar = self.logvar_fc(h)

        # re-parameterization trick
        z = self.reparameterize(mean, logvar)

        x_recon = self.decode(z)

        loss = self.loss(x, x_recon, mean, logvar)

        return h, x_recon, loss

    def loss(self, x, x_recon, mean, logvar):
        # reconstruction loss
        recon_loss = F.mse_loss(x, x_recon)
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return recon_loss + kl_loss
