"""Learnable parcellation: let the model decide which regions form a network.

Still a stub. It is the one model here that would change the multiverse rather
than sit inside it:
every specification in `acnets.multiverse` takes the region-to-network mapping
as given and varies what is computed on top of it, so a model that learns the
mapping is asking a question the grid cannot.

Notes:
  - Step1: The model receives region-level timeseries
    and fits network-level timeseries, one per network.
    Each region only belongs to one network.
  - Step2: The model receives network-level timeseries
    and calculates similarity matrix between them, e.g. cosine similarity.
  - Step3: The model receives similarity matrix and flattens it and only
    keep the upper triangular part as it's symmetric.
  - Step4: The flatten similarity vector is then fed to a multi-layer dense
    later to predict the target variable, i.e., AVGP or NVGP.

  - Normalization: region-level timeseries are normalized by min-max scaling.

The hard part is step 1. A hard assignment of regions to networks is not
differentiable, so it has to be relaxed -- a softmax over networks per region,
annealed towards one-hot, is the obvious route -- and the relaxation is what
decides whether the learned parcellation means anything or just launders the
classifier's own preferences.
"""

from torch import nn


class HierarchicalTemporalModel(nn.Module):

    def __init__(self, n_regions: int, n_networks: int = 6, **kwargs):
        super().__init__()
        self.n_regions = n_regions
        self.n_networks = n_networks

    def forward(self, x):
        raise NotImplementedError('HierarchicalTemporalModel.forward()')

    def compute_loss(self, batch, device):
        raise NotImplementedError('HierarchicalTemporalModel.compute_loss()')
