import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, n_embeddings):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, n_embeddings),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self, n_embeddings, n_outputs):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(n_embeddings, n_outputs * 2),
            nn.ReLU(),
            nn.Linear(n_outputs * 2, n_outputs),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.decode(x)
        return out


class VAE(nn.Module):
    def __init__(self, input_size, n_embeddings):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.encode = Encoder(input_size, n_embeddings)
        self.decode = Decoder(n_embeddings, input_size)

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
