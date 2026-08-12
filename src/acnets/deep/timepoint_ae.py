"""Per-timepoint autoencoder: a learnable connectivity representation.

Why this architecture, given that a sequence autoencoder was tried first and
rejected: on z-scored network-scale resting timeseries, held-out reconstruction
of a Seq2SeqAE plateaued at 1.004 relative to predicting the window mean — the
temporal unrolling learned nothing (`outputs/pretrain/lemon_seq2seq_*.json`).
What resting windows reliably carry is not a reproducible waveform but a
cross-channel covariance structure, which is exactly what every connectivity
statistic in this project estimates. So the encoder is applied per timepoint:
it compresses the 6 network amplitudes through a bottleneck, and can only
reconstruct well by learning how networks co-fluctuate. Phase alignment across
windows stops mattering, which is what sank the sequence model.

The classification features it induces are the latent second moments: encode a
subject's run to a latent series (n_timepoints x k), then take latent means and
the upper triangle of the latent covariance. That is connectivity computed in a
*learned* basis rather than the atlas basis — a small, testable version of the
learnable-parcellation idea in `hierarchical_temporal.py` and of the proposal's
"learned alignment subspace".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TimepointAE(nn.Module):
    """Bottleneck autoencoder over single timepoints.

    forward accepts (batch, n_channels) or (batch, n_timepoints, n_channels);
    the encoder is shared across timepoints in the latter case.
    """

    def __init__(self, n_channels: int, latent: int = 4, hidden: int = 16):
        super().__init__()
        if latent >= n_channels:
            raise ValueError('latent must be a bottleneck: '
                             f'latent={latent} >= n_channels={n_channels}')
        self.n_channels = n_channels
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z

    def compute_loss(self, batch, device):
        x = batch[0].to(device)
        x_recon, _ = self(x)
        loss = nn.functional.mse_loss(x_recon, x)
        # windows are z-scored, so variance ~1 and this reads as "fraction of
        # variance left unexplained"; below 1 means covariance was learned
        return loss, {'relative_error': float(loss.detach()) / float(x.var())}

    @torch.no_grad()
    def latent_series(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_timepoints, n_channels) -> (batch, n_timepoints, latent)."""
        return self.encoder(x)


def latent_summary(z: torch.Tensor) -> torch.Tensor:
    """Pool a latent series into classification features.

    (batch, n_timepoints, k) -> (batch, k + k*(k+1)/2): per-latent temporal
    means, then the upper triangle (with diagonal) of the latent temporal
    covariance. The covariance block is the learned-basis analogue of the
    connectivity features the classic specifications use, which is what makes
    the primary comparable in kind, if not in vocabulary, to the grid.
    """
    if z.ndim != 3:
        raise ValueError(f'expected (batch, time, latent), got {tuple(z.shape)}')
    mean = z.mean(dim=1)
    centred = z - mean.unsqueeze(1)
    cov = torch.einsum('btk,btl->bkl', centred, centred) / (z.shape[1] - 1)
    rows, cols = np.triu_indices(z.shape[2])
    return torch.cat([mean, cov[:, rows, cols]], dim=1)


class LatentConnectivityClassifier(nn.Module):
    """Pretrained encoder + linear head over latent summary features.

    Two-phase fine-tuning is driven by the caller (freeze/unfreeze via
    `set_encoder_frozen`), keeping this a plain module.
    """

    def __init__(self, autoencoder: TimepointAE, n_classes: int = 2):
        super().__init__()
        self.autoencoder = autoencoder
        k = autoencoder.latent
        self.n_features = k + k * (k + 1) // 2
        self.head = nn.Linear(self.n_features, n_classes)

    def set_encoder_frozen(self, frozen: bool) -> None:
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(not frozen)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return latent_summary(self.autoencoder.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
