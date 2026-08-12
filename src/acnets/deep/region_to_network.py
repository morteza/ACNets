"""Region connectivity -> learned network connectivity -> label, end to end.

A stack of layers with a named middle. Region-by-region connectivity goes in, a
learned region-to-network assignment pools it into a network-by-network matrix,
a few dense layers read that matrix, and a single logit comes out. One loss, one
backward pass, every parameter trained jointly -- including the assignment, so
the network definition is learned rather than taken from an atlas.

    C in R^(R x R)          region connectivity, one atlas, one statistic
      |  S = softmax(S_logits / tau), R x K       LEARNED assignment
      |  N = S^T C S / (s s^T)                    pooling, no free parameters
      v
    N in R^(K x K)          learned network connectivity   <- the middle layer
      |  upper triangle -> K(K+1)/2 values
      |  few dense layers
      v
    logit -> AVGP / NVGP

**Why the first stage is structured rather than dense.** The obvious reading of
"a few linear layers from region connectivity" is `Linear(n_region_edges,
hidden)`. Dosenbach 2010 has 160 regions and so 12,880 edges; at hidden = 28
that first layer alone is 360,640 parameters against 24 training participants.
The structured form learns a region-to-network assignment instead -- 160 x 7 =
1,120 parameters, and only 7 free values per region -- and is still a linear map
from region connectivity to network connectivity, just a low-rank one whose rank
is the thing being learned. It is the same reduction the atlas performs, with the
assignment estimated rather than assumed.

**The relaxation is where this can go wrong, and the project's own stub said so
first:** a hard region-to-network assignment is not differentiable, so it is
relaxed to a softmax with temperature `tau`, annealed towards one-hot during
training. `multimodal.models.hierarchical_temporal` warned that "the relaxation
is what decides whether the learned parcellation means anything or just launders
the classifier's own preferences", and the check for that is not internal to the
model: it is whether the learned assignment survives a permuted-label refit, and
whether it agrees with the atlas it was initialised from.

**Initialisation carries the inductive bias.** With `init_assignment` the
softmax logits start at the atlas's own region-to-network map, so the model
begins where the parcellation says and pays a gradient cost to move. That is a
prior, not a constraint, and `assignment_drift` measures how far it travelled.
Starting from random is available and is the honest ablation: if the learned
assignment only works when initialised from the atlas, it is the atlas doing the
work.
"""

from __future__ import annotations

import torch
from torch import nn


class RegionToNetworkMLP(nn.Module):
    """Learned parcellation into a network-connectivity bottleneck, then a head.

    Parameters
    ----------
    n_regions
        Regions in the atlas this instance reads.
    n_networks
        Width of the learned network layer, K. The mid layer has K(K+1)/2
        values.
    hidden
        Sizes of the dense layers after the bottleneck. Two small layers is
        the default; at n = 32 this part is where capacity is cheapest and
        least useful.
    tau
        Softmax temperature for the assignment. Annealed towards `tau_final`
        by `anneal`, so training ends closer to a hard parcellation than it
        starts.
    init_assignment
        (n_regions, n_networks) atlas assignment to start from, or None for a
        random start.
    """

    def __init__(self,
                 n_regions: int,
                 n_networks: int = 7,
                 hidden: tuple[int, ...] = (16, 8),
                 tau: float = 1.0,
                 tau_final: float = 0.1,
                 dropout: float = 0.0,
                 init_assignment: torch.Tensor | None = None):
        super().__init__()
        self.n_regions = n_regions
        self.n_networks = n_networks
        self.tau = tau
        self.tau_final = tau_final

        if init_assignment is not None:
            if init_assignment.shape != (n_regions, n_networks):
                raise ValueError(
                    f'init_assignment {tuple(init_assignment.shape)} does not '
                    f'match ({n_regions}, {n_networks})')
            # log of a smoothed indicator: the atlas assignment is the mode,
            # not a hard constraint
            smoothed = init_assignment * 3.0 - 1.5
            self.assignment_logits = nn.Parameter(smoothed.clone().float())
            self.register_buffer('initial_assignment', init_assignment.float())
        else:
            self.assignment_logits = nn.Parameter(
                torch.randn(n_regions, n_networks) * 0.1)
            self.register_buffer('initial_assignment',
                                 torch.zeros(n_regions, n_networks))

        n_edges = n_networks * (n_networks + 1) // 2
        # Normalising the bottleneck is not cosmetic. The pooled values inherit
        # the scale of the connectivity statistic and the network sizes, and on
        # real data that lands far outside tanh's responsive range: the first
        # activation saturates, gradients vanish, and the model reports a
        # training accuracy near chance with an attribution of order 1e-23 --
        # which reads as "no signal" when it is actually "no gradient".
        # BatchNorm's statistics are updated only under model.train(), which
        # here is only ever called on the training fold.
        self.bottleneck_norm = nn.BatchNorm1d(n_edges)
        layers: list[nn.Module] = []
        width = n_edges
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

        rows, cols = torch.triu_indices(n_networks, n_networks)
        self.register_buffer('triu_rows', rows)
        self.register_buffer('triu_cols', cols)

    def anneal(self, progress: float) -> None:
        """Move tau from its initial value towards `tau_final`.

        `progress` runs 0 to 1 over training. Annealing sharpens the assignment
        so the model ends near a hard parcellation, which is what makes the
        learned map readable as a parcellation rather than as a dense mixing
        matrix.
        """
        self.tau = self.tau + (self.tau_final - self.tau) * min(max(progress, 0.0), 1.0)

    def assignment(self) -> torch.Tensor:
        """Soft region-to-network assignment, rows summing to one."""
        return torch.softmax(self.assignment_logits / max(self.tau, 1e-3), dim=-1)

    def network_connectivity(self, connectivity: torch.Tensor) -> torch.Tensor:
        """The middle layer: (batch, K, K) learned network connectivity."""
        assignment = self.assignment()
        # two matmuls rather than one einsum: (S^T C) S costs b*K*R*R + b*K*R*K
        # against b*R*R*K*K for the naive contraction, which is a factor of
        # R/K -- about 20 at Dosenbach's 160 regions and 7 networks.
        left = torch.matmul(assignment.T, connectivity)      # (b, K, R)
        pooled = torch.matmul(left, assignment)              # (b, K, K)
        sizes = assignment.sum(dim=0)
        return pooled / torch.outer(sizes, sizes).clamp(min=1e-6)

    def forward(self, connectivity: torch.Tensor) -> torch.Tensor:
        """(batch, R, R) region connectivity -> (batch,) logits."""
        network = self.network_connectivity(connectivity)
        flat = self.bottleneck_norm(network[:, self.triu_rows, self.triu_cols])
        return self.head(flat).squeeze(-1)

    @torch.no_grad()
    def assignment_drift(self) -> float:
        """Fraction of regions whose hard assignment moved from the atlas.

        Zero means the model kept the parcellation it was given; one means it
        replaced it entirely. Reported because a learned parcellation that
        matches the atlas and a learned parcellation that ignores it support
        very different claims, and the accuracy alone cannot tell them apart.
        """
        if not self.initial_assignment.any():
            return float('nan')
        started = self.initial_assignment.argmax(dim=-1)
        ended = self.assignment().argmax(dim=-1)
        return float((started != ended).float().mean())

    def edge_attribution(self, connectivity: torch.Tensor) -> torch.Tensor:
        """Mean |gradient x input| at the learned network-connectivity layer.

        Taken at the middle layer rather than at the input, because that is the
        layer with a name: its K(K+1)/2 values are network pairs, and they are
        the units the rest of this project reasons about. Section 3.4.3 applies
        unchanged -- an attribution from a model that does not decode has to be
        tested against a permuted-label refit before it means anything.
        """
        network = self.network_connectivity(connectivity)
        flat = self.bottleneck_norm(
            network[:, self.triu_rows, self.triu_cols]).detach().requires_grad_(True)
        logits = self.head(flat).squeeze(-1)
        (gradient,) = torch.autograd.grad(logits.sum(), flat)
        return (gradient * flat).abs().mean(dim=0).detach()


class MultiAtlasRegionToNetwork(nn.Module):
    """One learned parcellation per atlas, all pooling into one network space.

    The single-atlas model above learns a map from that atlas's regions to K
    networks. With several atlases the interesting constraint is that they map
    into the *same* K networks: each atlas keeps its own assignment matrix,
    because it has its own regions, but the network-connectivity middle layer
    and everything after it are shared. So the head is trained on nine
    descriptions of the same seven networks rather than on nine unrelated
    feature sets, and adding an atlas adds data to the head rather than
    parameters.

        C_a in R^(R_a x R_a)     one region connectivity per atlas
          |  S_a = softmax(logits_a / tau)      one assignment per atlas
          v
        N_a in R^(K x K)         all atlases in the SAME network space
          |  per-atlas normalisation, then attention over atlases
          v
        shared head -> logit

    Initialising each `logits_a` from `canonical_networks.CROSSWALK` gives every
    atlas a starting map into the same vocabulary. Regions the crosswalk cannot
    place -- Gordon's `RetrosplenialTemporal`, Schaefer's `Limbic`, a quarter of
    Gordon's regions in total -- start uniform and are free to be learned, which
    is the one part of the parcellation this model can discover rather than
    inherit.

    The parameter count is dominated by the assignments: sum of R_a x K, which
    is about 9,500 across nine atlases against 24 training participants. That
    ratio is why `drift` is reported against a permuted-label null rather than
    read directly -- an assignment that moves the same distance whether or not
    the labels are real has learned nothing.
    """

    def __init__(self,
                 region_counts: list[int],
                 n_networks: int = 7,
                 hidden: tuple[int, ...] = (16, 8),
                 tau: float = 1.0,
                 tau_final: float = 0.1,
                 dropout: float = 0.0,
                 init_assignments: list[torch.Tensor] | None = None):
        super().__init__()
        self.n_atlases = len(region_counts)
        self.n_networks = n_networks
        self.tau = tau
        self.tau_final = tau_final

        logits, initials, presence = [], [], []
        for index, n_regions in enumerate(region_counts):
            if init_assignments is not None:
                atlas = init_assignments[index].float()
                # Which canonical networks this atlas can populate at all.
                # Gordon has no cerebellum and Dosenbach 2010 no dorsal
                # attention; without this mask the regions the crosswalk could
                # not place would spread uniformly into those networks and
                # manufacture a value for an edge the atlas cannot measure.
                present = atlas.sum(dim=0) > 0
                placed = atlas.sum(dim=-1, keepdim=True) > 0
                start = torch.where(placed, atlas * 3.0 - 1.5,
                                    torch.zeros_like(atlas))
                # unplaced regions are free, but only over networks that exist
                start = start.masked_fill(~present.unsqueeze(0), -1e4)
                initials.append(atlas)
            else:
                present = torch.ones(n_networks, dtype=torch.bool)
                start = torch.randn(n_regions, n_networks) * 0.1
                initials.append(torch.zeros(n_regions, n_networks))
            presence.append(present)
            logits.append(nn.Parameter(start.clone()))

        self.assignment_logits = nn.ParameterList(logits)
        for index, initial in enumerate(initials):
            self.register_buffer(f'initial_{index}', initial)

        rows, cols = torch.triu_indices(n_networks, n_networks)
        # (n_atlases, n_edges): can this atlas measure this network pair
        edge_presence = torch.stack(
            [(present[rows] & present[cols]) for present in presence])
        self.register_buffer('edge_presence', edge_presence.float())

        n_edges = n_networks * (n_networks + 1) // 2
        self.bottleneck_norm = nn.ModuleList(
            nn.BatchNorm1d(n_edges) for _ in region_counts)
        self.atlas_logits = nn.Parameter(torch.zeros(self.n_atlases))

        layers: list[nn.Module] = []
        width = n_edges
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

        rows, cols = torch.triu_indices(n_networks, n_networks)
        self.register_buffer('triu_rows', rows)
        self.register_buffer('triu_cols', cols)

    def anneal(self, progress: float) -> None:
        self.tau = self.tau + (self.tau_final - self.tau) * min(max(progress, 0.0), 1.0)

    def assignment(self, index: int) -> torch.Tensor:
        return torch.softmax(self.assignment_logits[index] / max(self.tau, 1e-3),
                             dim=-1)

    def network_connectivity(self, connectivity: torch.Tensor,
                             index: int) -> torch.Tensor:
        assignment = self.assignment(index)
        pooled = torch.matmul(torch.matmul(assignment.T, connectivity), assignment)
        sizes = assignment.sum(dim=0)
        return pooled / torch.outer(sizes, sizes).clamp(min=1e-6)

    def bottleneck(self, connectivities: list[torch.Tensor]) -> torch.Tensor:
        """(batch, n_atlases, n_edges) normalised network connectivity."""
        stacked = []
        for index, connectivity in enumerate(connectivities):
            network = self.network_connectivity(connectivity, index)
            flat = network[:, self.triu_rows, self.triu_cols]
            stacked.append(self.bottleneck_norm[index](flat)
                           * self.edge_presence[index])
        return torch.stack(stacked, dim=1)

    def pool_atlases(self, stacked: torch.Tensor) -> torch.Tensor:
        """Attention over atlases, renormalised per edge by who can measure it.

        Cerebellum exists in three of nine atlases and dorsal attention in
        eight. Summing attention-weighted contributions without accounting for
        that gives the sparse edges a fraction of the weight the dense ones
        get, purely because fewer atlases populate them -- an artefact of atlas
        coverage that would read as those networks mattering less. Dividing by
        the weight actually available for each edge puts every network pair on
        the same footing, which is the whole point of a shared vocabulary.
        """
        weights = torch.softmax(self.atlas_logits, dim=0).view(1, -1, 1)
        available = weights * self.edge_presence.unsqueeze(0)
        return ((stacked * available).sum(dim=1)
                / available.sum(dim=1).clamp(min=1e-6))

    def forward(self, connectivities: list[torch.Tensor]) -> torch.Tensor:
        return self.head(self.pool_atlases(self.bottleneck(connectivities))
                         ).squeeze(-1)

    @torch.no_grad()
    def atlas_weights(self) -> torch.Tensor:
        return torch.softmax(self.atlas_logits, dim=0)

    @torch.no_grad()
    def assignment_drift(self) -> list[float]:
        """Per-atlas fraction of regions whose network changed from the crosswalk.

        Only regions the crosswalk placed are counted: the unplaced ones start
        uniform, so any assignment they end with is learned rather than moved,
        and counting them would report drift for regions that never had a
        starting position.
        """
        drifts = []
        for index in range(self.n_atlases):
            initial = getattr(self, f'initial_{index}')
            placed = initial.sum(dim=-1) > 0
            if not placed.any():
                drifts.append(float('nan'))
                continue
            started = initial[placed].argmax(dim=-1)
            ended = self.assignment(index)[placed].argmax(dim=-1)
            drifts.append(float((started != ended).float().mean()))
        return drifts

    def edge_attribution(self, connectivities: list[torch.Tensor]) -> torch.Tensor:
        """Mean |gradient x input| over the shared network edges.

        One vector in the canonical vocabulary, not one per atlas: the point of
        the shared middle layer is that the atlases are describing the same
        seven networks, so their attributions are commensurable and are pooled
        by the same attention weights the prediction uses.
        """
        stacked = self.bottleneck(connectivities).detach().requires_grad_(True)
        logits = self.head(self.pool_atlases(stacked)).squeeze(-1)
        (gradient,) = torch.autograd.grad(logits.sum(), stacked)
        return (gradient * stacked).abs().mean(dim=(0, 1)).detach()
