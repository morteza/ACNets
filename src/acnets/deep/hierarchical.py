"""One model over every atlas and every connectivity statistic at once.

The architecture the project's stub (`multimodal.models.hierarchical_temporal`)
described, built the way n = 32 allows:

    region connectivity            one (atlas x statistic) matrix per view
      |  fixed region-to-network grouping, zero learned parameters
      v
    canonical network connectivity 28 network pairs, same vocabulary for every
      |                            view, supplied by `canonical_networks`
      |  encoder, weights SHARED across all views
      v
    per-view embedding
      |  learned view weights, softmax over views
      v
    pooled embedding -> logit -> AVGP / NVGP

**Two decisions carry the whole design, and both are about parameter count
rather than about expressiveness.**

The region-to-network map is fixed, not learned. Learning it means a
regions x networks assignment per atlas -- 960 free parameters for Dosenbach
alone, against 24 training participants -- and the stub already anticipated what
that produces: "the relaxation is what decides whether the learned parcellation
means anything or just launders the classifier's own preferences". Fixing it
spends zero degrees of freedom and makes the grouping an inductive bias in the
strict sense, a constraint carrying information from outside the sample.

The encoder is shared across views. Nine atlases by six statistics is 54 views;
a separate encoder per view would be 54 x 28 parameters and would simply fit each
view's noise. One shared encoder over a common vocabulary means adding an atlas
adds *data* rather than parameters, which is the only way a multi-atlas model is
estimable at this sample size. It also makes the model's attribution directly
comparable across atlases, which is the property `run_agreement.py` says it
cannot currently measure.

Default size is 28 x d + V + d + 1 parameters: 85 at d = 1 with 54 views. That is
the honest ceiling on complexity here, and the model is closer to a structured
linear model than to a deep network. Section 3.4.2 is the reason to expect this
not to matter -- classifier family moves 3-5% of accuracy variance -- and section
3.7 is the reason to expect no architecture to clear the corrected null.
"""

from __future__ import annotations

import torch
from torch import nn


class HierarchicalConnectivityModel(nn.Module):
    """Shared-encoder attention pooling over (atlas, statistic) views.

    Parameters
    ----------
    n_views
        Number of (atlas, statistic) inputs.
    n_edges
        Canonical network pairs per view; 28 for the seven-network vocabulary.
    embedding_dim
        Width of the shared encoder. 1 makes the model a weighted linear
        combination of views and is the default, because everything in this
        project says the leverage is not in model capacity.
    view_attention
        'learned' gives one global weight per view, softmaxed -- interpretable
        as "which atlas and statistic does the model use", and the reason to
        prefer it. 'content' scores each view from its own embedding, which is
        per-participant and more expressive but has no single reportable value.
    """

    def __init__(self,
                 n_views: int,
                 n_edges: int = 28,
                 embedding_dim: int = 1,
                 view_attention: str = 'learned',
                 dropout: float = 0.0):
        super().__init__()
        if view_attention not in ('learned', 'content'):
            raise ValueError(f'unknown view_attention: {view_attention!r}')

        self.n_views = n_views
        self.n_edges = n_edges
        self.view_attention = view_attention

        self.encoder = nn.Linear(n_edges, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(embedding_dim, 1)

        if view_attention == 'learned':
            self.view_logits = nn.Parameter(torch.zeros(n_views))
        else:
            self.query = nn.Linear(embedding_dim, 1, bias=False)

    def view_weights(self, embeddings: torch.Tensor,
                     view_mask: torch.Tensor) -> torch.Tensor:
        """Softmax over views, with absent views excluded rather than zeroed."""
        if self.view_attention == 'learned':
            scores = self.view_logits.unsqueeze(0).expand(embeddings.shape[0], -1)
        else:
            scores = self.query(embeddings).squeeze(-1)
        scores = scores.masked_fill(~view_mask, float('-inf'))
        return torch.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor,
                edge_mask: torch.Tensor,
                view_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (batch, n_views, n_edges) canonical network connectivity.
        edge_mask
            (n_views, n_edges) True where the network pair exists in that
            view's atlas. Unpopulated pairs are *missing*, not zero -- Dosenbach
            2010 has no dorsal attention network and every atlas but three has
            no cerebellum -- so they are zeroed after standardisation rather
            than allowed to contribute a spurious value.
        view_mask
            (batch, n_views) True where the view is available. Defaults to all
            available; present so the model can be fitted on subsets without
            changing shape.

        Returns
        -------
        (batch,) logits.
        """
        masked = x * edge_mask.unsqueeze(0)
        embeddings = self.encoder(masked)
        embeddings = self.dropout(embeddings)

        if view_mask is None:
            view_mask = torch.ones(x.shape[0], self.n_views, dtype=torch.bool,
                                   device=x.device)
        weights = self.view_weights(embeddings, view_mask)
        pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def reported_view_weights(self) -> torch.Tensor | None:
        """Global view weights, when they exist as a single reportable vector."""
        if self.view_attention != 'learned':
            return None
        return torch.softmax(self.view_logits, dim=-1)

    def attribution(self, x: torch.Tensor,
                    edge_mask: torch.Tensor,
                    view_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean |gradient x input| over the canonical edges, pooled across views.

        The same channel the multiverse's neural specifications use (section
        6.7), so the numbers are comparable with those. Read the caveat in
        section 3.4.3 before interpreting it: an attribution from a model that
        does not decode is testable only against a permuted-label refit, and
        only when the grid it sits in varies the model family.
        """
        x = x.clone().requires_grad_(True)
        logits = self.forward(x, edge_mask, view_mask)
        (gradient,) = torch.autograd.grad(logits.sum(), x)
        contribution = (gradient * x).abs()             # (batch, views, edges)
        weight = edge_mask.unsqueeze(0).float()
        total = (contribution * weight).sum(dim=(0, 1))
        counts = weight.expand_as(contribution).sum(dim=(0, 1))
        return (total / counts.clamp(min=1.0)).detach()
