"""Shared-latent fusion over the multiverse's own views.

The specification grid treats connectivity statistics as competing analysis
paths; this model treats them as complementary measurements of one subject.
Each view — a connectivity vector per statistic, the structural volume vector,
the behavioural summary — gets its own encoder into a shared latent space;
available views' latents are averaged; a linear head classifies. Exploratory
by declaration: the confirmatory primary is the latent-connectivity model
(`prereg_primary.md`), and this one is reported descriptively.

Missing modalities are first-class, not an error: behavioural records exist for
25 of 32 subjects and native anatomy for 31, so any fusion model for this
dataset must accept arbitrary subsets. During training, *modality dropout*
randomly hides available views, which both regularises and trains the model to
operate under exactly the missingness it will meet at inference.
"""

from __future__ import annotations

import torch
from torch import nn


class MultiViewClassifier(nn.Module):
    """Per-view encoders -> masked mean fusion -> linear head.

    Parameters
    ----------
    view_dims
        Mapping of view name to input dimensionality, e.g.
        {'correlation': 21, 'tangent': 21, 'structure': 267, 'behaviour': 3}.
    """

    def __init__(self, view_dims: dict[str, int], latent: int = 8,
                 hidden: int = 32, n_classes: int = 2,
                 modality_dropout: float = 0.3):
        super().__init__()
        if not view_dims:
            raise ValueError('need at least one view')
        self.view_names = tuple(view_dims)
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, latent))
            for name, dim in view_dims.items()})
        self.head = nn.Linear(latent, n_classes)

    def fuse(self, views: dict[str, torch.Tensor],
             mask: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Masked mean of encoded views.

        `mask[name]` is a (batch,) float tensor, 1 where the view is present.
        During training, present views are additionally dropped at random --
        except that at least one view is always kept per sample, enforced by
        re-instating a random present view where the dropout would have
        silenced everything.
        """
        first = next(iter(views.values()))
        batch = first.shape[0]
        latents, weights = [], []

        keep = {}
        for name in self.view_names:
            if name not in views:
                continue
            present = (mask[name] if mask is not None and name in mask
                       else torch.ones(batch, device=first.device))
            if self.training and self.modality_dropout > 0:
                drop = (torch.rand(batch, device=first.device)
                        < self.modality_dropout).float()
                keep[name] = present * (1.0 - drop)
            else:
                keep[name] = present

        # never silence every view of a sample
        total = torch.stack(list(keep.values())).sum(dim=0)
        for name in keep:
            present = (mask[name] if mask is not None and name in mask
                       else torch.ones(batch, device=first.device))
            keep[name] = torch.where(total > 0, keep[name], present)

        for name, kept in keep.items():
            latents.append(self.encoders[name](views[name]) * kept.unsqueeze(1))
            weights.append(kept)

        stacked = torch.stack(latents).sum(dim=0)
        denominator = torch.stack(weights).sum(dim=0).clamp(min=1.0)
        return stacked / denominator.unsqueeze(1)

    def forward(self, views: dict[str, torch.Tensor],
                mask: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        return self.head(self.fuse(views, mask))
