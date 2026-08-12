"""A denoising diffusion model over connectivity vectors, for normative scoring.

The point is not to generate connectomes. It is to learn, from 220 external
participants who were never labelled, what a *typical* resting connectivity
vector looks like -- and then ask, for each participant in the target sample,
how atypical theirs is.

Why this is a different quantity from everything else in the multiverse. The six
connectivity statistics are all functions of one covariance matrix, so a
classifier over any of them sees the same information rearranged. A normative
deviation score is not a rearrangement: it is the target participant's position
relative to a density estimated on other people's data. Two participants can
have identical CON-SMN coupling and different deviation scores, because the
score depends on the rest of the vector and on how the external population
covaries. That is the one respect in which it can carry information the grid
cannot.

**Architecture.** Standard DDPM with an epsilon-prediction objective, over a
vector rather than an image, so the denoiser is an MLP with a sinusoidal
timestep embedding rather than a U-Net. At 21 dimensions and ~1,100 training
windows a convolutional backbone would be silly; the data are a short vector.

**The deviation score.** After training, a participant's vector x0 is scored by
how badly the model denoises it, averaged over noise levels and noise draws:

    deviation_j = E_{t, eps} | eps_pred(x_t, t)_j - eps_j |

Per dimension, not just as a scalar. That matters twice over: the per-dimension
vector is what the classifier consumes, and it lives in exactly the same
21-edge vocabulary as every attribution in the paper, so it drops into the
existing agreement analysis without a translation layer.

Averaging over many (t, eps) draws is not optional. A single draw is dominated
by which noise happened to be sampled, and the resulting score is mostly noise
about noise. `n_draws` controls it and the default is deliberately generous.

**Leakage.** Nothing here ever receives a label; `fit` takes an array of
vectors. Training on an external dataset and scoring the target is therefore
leakage-free in the same sense as `acnets.multiverse.transfer`, and asserted the
same way by a test that permutes the target labels and requires bit-identical
scores.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def cosine_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative alphas under the cosine schedule (Nichol and Dhariwal 2021).

    Preferred over the original linear schedule because at the small step counts
    that suffice for low-dimensional data the linear schedule destroys the
    signal too early, leaving most timesteps uninformative.
    """
    steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = (f / f[0]).clamp(1e-8, 1.0)
    return alpha_bar[1:].float()


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion timestep, then a small MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        angles = t.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dim:            # odd dim
            embedding = torch.cat(
                [embedding, torch.zeros(len(t), 1, device=t.device)], dim=1)
        return self.net(embedding)


class VectorDiffusion(nn.Module):
    """DDPM over fixed-length vectors, with an MLP epsilon-predictor.

    Parameters
    ----------
    n_features
        Length of the vector being modelled, e.g. 21 network edges.
    hidden, n_layers
        Denoiser width and depth.
    n_steps
        Diffusion steps. Low-dimensional data needs far fewer than an image
        model; 200 is ample and keeps scoring cheap.
    """

    def __init__(self, n_features: int, hidden: int = 128, n_layers: int = 3,
                 n_steps: int = 200, time_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.n_steps = n_steps
        self.hparams = {'n_features': n_features, 'hidden': hidden,
                        'n_layers': n_layers, 'n_steps': n_steps,
                        'time_dim': time_dim}

        self.time = TimestepEmbedding(time_dim)
        layers: list[nn.Module] = [nn.Linear(n_features + time_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, n_features)]
        self.net = nn.Sequential(*layers)

        alpha_bar = cosine_schedule(n_steps)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus', (1 - alpha_bar).sqrt())

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicted noise in `x_t`."""
        return self.net(torch.cat([x_t, self.time(t)], dim=1))

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        return (self.sqrt_alpha_bar[t][:, None] * x0
                + self.sqrt_one_minus[t][:, None] * noise)

    def compute_loss(self, batch, device):
        """Plugs into `acnets.deep.Trainer`."""
        x0 = batch[0].to(device)
        t = torch.randint(0, self.n_steps, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        predicted = self(self.add_noise(x0, t, noise), t)
        loss = nn.functional.mse_loss(predicted, noise)
        return loss, {'loss': float(loss.detach())}

    @torch.no_grad()
    def deviation(self, x: np.ndarray, n_draws: int = 64, seed: int = 0,
                  max_t: float = 1.0) -> np.ndarray:
        """Per-dimension normative deviation for each row of `x`.

        Returns (n_samples, n_features): the mean absolute error of the model's
        noise prediction, per dimension, averaged over `n_draws` draws of
        (timestep, noise).

        `max_t` restricts the timesteps sampled to the first fraction of the
        schedule. At high noise levels every input looks alike and the score
        stops discriminating, so limiting to the low-noise end sharpens it;
        1.0 uses the whole schedule.
        """
        self.eval()
        device = next(self.parameters()).device
        x0 = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        ceiling = max(1, int(self.n_steps * max_t))

        total = torch.zeros_like(x0)
        for _ in range(n_draws):
            t = torch.randint(0, ceiling, (len(x0),), generator=generator).to(device)
            noise = torch.randn(x0.shape, generator=generator).to(device)
            predicted = self(self.add_noise(x0, t, noise), t)
            total += (predicted - noise).abs()
        return (total / n_draws).cpu().numpy()


class GaussianNormative:
    """Mahalanobis deviation from an external population. The control condition.

    A normative model does not have to be deep. This is the classical version --
    fit a mean and a shrunk covariance on the external data, score the target by
    Mahalanobis distance -- and it exists so the diffusion model has to earn its
    complexity. If the two score equally, the extra capacity bought nothing, and
    that is worth knowing before a diffusion model goes into a paper.
    """

    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage

    def fit(self, x: np.ndarray) -> GaussianNormative:
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        covariance = np.cov(x, rowvar=False)
        covariance = np.atleast_2d(covariance)
        # Ledoit-Wolf style shrinkage toward a scaled identity, so the inverse
        # exists even when the external sample is small relative to the vector
        target = np.eye(len(covariance)) * np.trace(covariance) / len(covariance)
        self.covariance_ = (1 - self.shrinkage) * covariance + self.shrinkage * target
        self.precision_ = np.linalg.pinv(self.covariance_)
        return self

    def deviation(self, x: np.ndarray) -> np.ndarray:
        """Per-dimension deviation, in the same shape the diffusion model returns."""
        centred = np.asarray(x, dtype=float) - self.mean_
        # per-dimension contribution to the Mahalanobis quadratic form, so the
        # output is comparable in shape and interpretation to the diffusion score
        return np.abs(centred * (centred @ self.precision_))

    def distance(self, x: np.ndarray) -> np.ndarray:
        """Scalar Mahalanobis distance per row."""
        centred = np.asarray(x, dtype=float) - self.mean_
        return np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', centred,
                                               self.precision_, centred)))
