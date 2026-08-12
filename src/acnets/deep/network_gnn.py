"""End-to-end graph network over canonical brain networks.

Connectivity in, AVGP/NVGP out, one loss, every parameter trained jointly. No
feature selection, no classical classifier at the end, no stage fitted
separately.

Each canonical network -- CON, FPN, SMN, DMN, VIS, DAN, CER -- is a node
carrying a learned embedding of `node_dim` values rather than a scalar. The
connectivity matrix supplies the edge weights between those nodes, and it is the
only place subject-specific information enters: the node embedding table is
shared across participants, so everything that distinguishes one participant
from another arrives through the graph's edges.

    E in R^(7 x node_dim)              learned node embeddings, shared
    A_v in R^(7 x 7)                   this participant's network connectivity,
                                       one graph per (atlas, statistic) view
      |  L rounds of message passing:  H <- act(A_hat H W + H U)
      v
    H_v in R^(7 x node_dim)            contextualised node states
      |  readout over nodes, then attention over views
      v
    logit -> AVGP / NVGP

**Why the node embedding is learned rather than derived.** A scalar per node
cannot distinguish "this network is strongly coupled to two others" from "this
network is the hub of the graph"; a vector can carry a role. The embedding is
free to learn what role each canonical network plays, and because the table is
shared across all views and all participants, it is estimated from every atlas at
once rather than per atlas.

**Parameter budget, stated because it is the binding constraint.** With
node_dim = 4 and two message-passing rounds the model has roughly 130
parameters against 24 training participants per fold; node_dim = 8 roughly
triples the message-passing weights. Section 3.7 puts the reliability ceiling at
0.72 against a family-wise threshold of 0.739, and section 3.8 records thirteen
classifier families, nested tuning, external pretraining, fusion, instance-level
learning and diffusion normative modelling all failing to move the result. This
architecture is not expected to clear the ceiling either, and the reason to build
it is the shared canonical vocabulary its attribution lives in, not the accuracy.

Training-set accuracy is reported alongside test accuracy for a specific reason:
the previous model in this project reached 0.375 test, below chance, which is
what a memorised training fold looks like. A gap that wide is a fact about
optimisation, not about brains, and it should be visible rather than inferred.
"""

from __future__ import annotations

import torch
from torch import nn


def symmetric_from_upper(values: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """(..., n_edges) upper triangle including diagonal -> (..., n, n)."""
    rows, cols = torch.triu_indices(n_nodes, n_nodes)
    shape = (*values.shape[:-1], n_nodes, n_nodes)
    matrix = values.new_zeros(shape)
    matrix[..., rows, cols] = values
    matrix[..., cols, rows] = values
    return matrix


class NetworkGraphClassifier(nn.Module):
    """Message passing over canonical networks, trained end to end.

    Parameters
    ----------
    n_views
        Number of (atlas, statistic) graphs per participant.
    n_nodes
        Canonical networks; 7 here.
    node_dim
        Width of each node's embedding. 4 to 8 is the range worth searching at
        this sample size; the default of 4 is the smaller end deliberately.
    n_layers
        Message-passing rounds. Two is enough for a 7-node graph -- every node
        reaches every other in one hop on a dense graph, so depth beyond two
        buys mixing rather than reach.
    readout
        'mean' averages node states; 'attention' learns which nodes to read,
        which gives a per-network importance vector for free but costs
        node_dim + 1 parameters.
    """

    def __init__(self,
                 n_views: int,
                 n_nodes: int = 7,
                 node_dim: int = 4,
                 n_layers: int = 2,
                 readout: str = 'attention',
                 view_attention: str = 'learned',
                 dropout: float = 0.0):
        super().__init__()
        if readout not in ('mean', 'attention'):
            raise ValueError(f'unknown readout: {readout!r}')
        if view_attention not in ('learned', 'content'):
            raise ValueError(f'unknown view_attention: {view_attention!r}')

        self.n_views = n_views
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.readout = readout
        self.view_attention = view_attention

        self.node_embedding = nn.Parameter(torch.randn(n_nodes, node_dim) * 0.1)
        self.message = nn.ModuleList(
            nn.Linear(node_dim, node_dim, bias=False) for _ in range(n_layers))
        self.selfloop = nn.ModuleList(
            nn.Linear(node_dim, node_dim, bias=False) for _ in range(n_layers))
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if readout == 'attention':
            self.node_query = nn.Linear(node_dim, 1, bias=False)
        if view_attention == 'learned':
            self.view_logits = nn.Parameter(torch.zeros(n_views))
        else:
            self.view_query = nn.Linear(node_dim, 1, bias=False)

        self.head = nn.Linear(node_dim, 1)

    @staticmethod
    def normalise(adjacency: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Symmetric degree normalisation, with absent networks removed.

        Connectivity can be negative (partial correlation, precision), so the
        degree is taken over absolute weights; without that a node whose
        positive and negative edges cancel would get an unbounded scaling.
        """
        adjacency = adjacency * node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        degree = adjacency.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adjacency / degree

    def forward(self, x: torch.Tensor,
                node_mask: torch.Tensor,
                view_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (batch, n_views, n_edges) canonical network connectivity, upper
            triangle including the diagonal.
        node_mask
            (n_views, n_nodes) True where that view's atlas populates the
            network. Dosenbach 2010 has no dorsal attention network and only
            three atlases have cerebellum, so absent nodes are removed from
            message passing and from the readout rather than passed as zeros.
        """
        batch = x.shape[0]
        adjacency = symmetric_from_upper(x, self.n_nodes)
        mask = node_mask.unsqueeze(0).float()
        normalised = self.normalise(adjacency, mask)

        states = self.node_embedding.expand(batch, self.n_views, -1, -1)
        for message, selfloop in zip(self.message, self.selfloop, strict=True):
            neighbours = torch.matmul(normalised, states)
            states = self.activation(message(neighbours) + selfloop(states))
            states = self.dropout(states)
        states = states * mask.unsqueeze(-1)

        if self.readout == 'attention':
            scores = self.node_query(states).squeeze(-1)
            scores = scores.masked_fill(~node_mask.unsqueeze(0).bool(),
                                        float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            pooled = (states * weights.unsqueeze(-1)).sum(dim=-2)
        else:
            counts = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooled = states.sum(dim=-2) / counts

        if view_mask is None:
            view_mask = torch.ones(batch, self.n_views, dtype=torch.bool,
                                   device=x.device)
        if self.view_attention == 'learned':
            view_scores = self.view_logits.unsqueeze(0).expand(batch, -1)
        else:
            view_scores = self.view_query(pooled).squeeze(-1)
        view_scores = view_scores.masked_fill(~view_mask, float('-inf'))
        view_weights = torch.softmax(view_scores, dim=-1)

        graph = (pooled * view_weights.unsqueeze(-1)).sum(dim=1)
        return self.head(graph).squeeze(-1)

    @torch.no_grad()
    def reported_view_weights(self) -> torch.Tensor | None:
        if self.view_attention != 'learned':
            return None
        return torch.softmax(self.view_logits, dim=-1)

    def attribution(self, x: torch.Tensor,
                    node_mask: torch.Tensor,
                    view_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean |gradient x input| over canonical network edges.

        End-to-end differentiability is what makes this cheap: the gradient of
        the logit with respect to the input connectivity runs through the
        message passing, the readout and the view attention in one backward
        pass, so the attribution describes the whole model rather than a final
        layer. Section 3.4.3 is the caveat -- test it against a permuted-label
        refit before reading it as anything about participants.
        """
        x = x.clone().requires_grad_(True)
        logits = self.forward(x, node_mask, view_mask)
        (gradient,) = torch.autograd.grad(logits.sum(), x)
        contribution = (gradient * x).abs()
        return contribution.mean(dim=(0, 1)).detach()

    def node_importance(self, x: torch.Tensor,
                        node_mask: torch.Tensor) -> torch.Tensor:
        """Gradient norm at each node's contextualised state.

        The node-level counterpart of the edge attribution, available because
        nodes carry vectors rather than scalars: it says which canonical
        networks the readout depends on, not merely which edges.
        """
        batch = x.shape[0]
        adjacency = symmetric_from_upper(x, self.n_nodes)
        mask = node_mask.unsqueeze(0).float()
        normalised = self.normalise(adjacency, mask)

        states = self.node_embedding.expand(batch, self.n_views, -1, -1)
        for message, selfloop in zip(self.message, self.selfloop, strict=True):
            neighbours = torch.matmul(normalised, states)
            states = self.activation(message(neighbours) + selfloop(states))
        states = (states * mask.unsqueeze(-1)).detach().requires_grad_(True)

        if self.readout == 'attention':
            scores = self.node_query(states).squeeze(-1)
            scores = scores.masked_fill(~node_mask.unsqueeze(0).bool(),
                                        float('-inf'))
            pooled = (states * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(-2)
        else:
            pooled = states.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        logits = self.head(pooled.mean(dim=1)).squeeze(-1)
        (gradient,) = torch.autograd.grad(logits.sum(), states)
        return gradient.norm(dim=-1).mean(dim=(0, 1)).detach()
