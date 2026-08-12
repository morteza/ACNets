import torch
from torch import nn
from torch.nn import functional as F

from .training import accuracy
from .vae import VAE


class MultiHeadModel(nn.Module):
    def __init__(self, n_regions, n_networks, n_embeddings=2):
        super().__init__()

        self.n_regions = n_regions
        self.n_networks = n_networks
        self.n_embeddings = n_embeddings

        # X1 (region-level timeseries)
        # self.x1_autoencoder = Seq2SeqAutoEncoder(n_regions, n_embeddings)
        # self.x1_head = nn.Sequential(
        #     nn.Linear(n_embeddings, n_embeddings)
        # )

        # X2 (region-level connectivity)
        # self.x2_head = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     VAE(n_regions * n_regions, n_embeddings)
        # )

        # X3 (network-level timeseries)
        # self.x3_autoencoder = Seq2SeqAutoEncoder(n_networks, n_embeddings)
        # self.x3_head = nn.Sequential(
        #     nn.Linear(n_embeddings, n_embeddings)
        # )

        # X4 (network-level connectivity)
        self.x4_head = nn.Sequential(
            # nn.Flatten(start_dim=1),
            VAE(n_networks * (n_networks - 1) // 2, n_embeddings)
        )

        # X5 (averaged network-level connectivity)
        self.x5_head = nn.Sequential(
            # nn.Flatten(start_dim=1),
            VAE(n_networks * (n_networks + 1) // 2, n_embeddings)
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

        h4, _x4_recon, loss_x4 = self.x4_head(x4)
        h5, _x5_recon, loss_x5 = self.x5_head(x5)

        loss_recon = loss_x4 + loss_x5

        h = torch.stack([h4, h5], dim=0).sum(dim=0)
        y = self.cls_head(h)

        return y, loss_recon

    def fine_tune(self, x1, x2, x3, x4, x5, y):
        raise NotImplementedError('Fine-tuning is not implemented yet.')

    def _vectorize(self, x4, x5):
        """Upper triangle of each connectivity matrix, flattened.

        x4 excludes the diagonal, x5 includes it. Both were repeated verbatim in
        four methods before; a difference introduced into one copy and not the
        others would have been invisible.
        """
        device = x4.device
        ones = torch.ones(self.n_networks, self.n_networks, device=device)
        x4 = x4[:, torch.triu(ones, diagonal=1) == 1]
        x5 = x5[:, torch.triu(ones, diagonal=0) == 1]
        return x4, x5

    def compute_loss(self, batch, device):
        x1, x2, x3, x4, x5, y = [t.to(device) for t in batch]
        x4, x5 = self._vectorize(x4, x5)

        y_hat, loss_recon = self(x1, x2, x3, x4, x5)

        beta = 1.  # reconstruction loss coefficient
        loss_recon = beta * loss_recon
        loss_cls = F.cross_entropy(y_hat, y)
        loss = loss_cls + loss_recon

        return loss, {
            'loss_cls': float(loss_cls),
            'loss_recon': float(loss_recon),
            'accuracy': accuracy(y_hat, y),
        }

    def enable_dropout(self):
        """Enables dropout during validation and test."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    @torch.no_grad()
    def validation_step(self, batch, device):
        x1, x2, x3, x4, x5, y = [t.to(device) for t in batch]
        x4, x5 = self._vectorize(x4, x5)

        y_hat, *_ = self(x1, x2, x3, x4, x5)
        loss_cls = F.cross_entropy(y_hat, y)

        return {
            'loss_cls': float(loss_cls),
            'accuracy': accuracy(y_hat, y),
            'dropout_accuracy': self.dropout_accuracy(batch, device),
        }

    test_step = validation_step

    @torch.no_grad()
    def dropout_accuracy(self, batch, device, n_repeats=10):
        """Accuracy with dropout left on, averaged over repeats.

        A cheap posterior-predictive check: if the prediction survives having
        random units removed, it does not rest on one path through the network.
        """
        x1, x2, x3, x4, x5, y = [t.to(device) for t in batch]
        x4, x5 = self._vectorize(x4, x5)

        was_training = self.training
        self.eval()
        self.enable_dropout()

        total = 0.0
        for _ in range(n_repeats):
            y_hat, *_ = self(x1, x2, x3, x4, x5)
            total += accuracy(y_hat, y)

        # restore whatever mode the caller was in; the Lightning version left
        # dropout enabled for everything that ran after it
        self.train(was_training)
        return total / n_repeats

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
