"""Feature blocks for the regional / within-network / between-network contrast.

Section 3.2 of the paper asks which part of the connectivity structure carries
the group signal. Answering it fairly means deriving all three blocks from the
*same* connectivity matrix, so that no block gets an estimation advantage, and
defining the within-network block over region pairs rather than over a matrix
diagonal.

The diagonal point is not pedantic. `acnets.pipeline.MultiScaleClassifier`
defines its within-network block as the diagonal of the network-scale matrix,
which is a constant for correlation and partial correlation (fixed at 1 by
construction) -- so that block carries no information for two of the six
statistics, and a comparison built on it silently answers a different question.
Its regional block is worse: the temporal mean of z-scored time courses, which
is zero up to numerical residue. These functions replace both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def regional_degree(matrices: np.ndarray) -> np.ndarray:
    """Weighted degree per region: mean absolute off-diagonal connectivity.

    Parameters
    ----------
    matrices
        (n_subjects, n_regions, n_regions) connectivity matrices.

    Returns
    -------
    (n_subjects, n_regions) array.

    Absolute values are used because degree is about how strongly a region
    participates, not about the sign of its couplings; a region with strong
    positive and strong negative edges is not weakly connected.
    """
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f'expected (subjects, n, n), got {matrices.shape}')
    n = matrices.shape[-1]
    off_diagonal = ~np.eye(n, dtype=bool)
    return np.stack([np.abs(m)[off_diagonal].reshape(n, n - 1).mean(axis=1)
                     for m in matrices])


def network_blocks(matrices: np.ndarray,
                   assignment: Sequence[str],
                   networks: Sequence[str],
                   ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Mean region-pair connectivity within each network and between each pair.

    Parameters
    ----------
    matrices
        (n_subjects, n_regions, n_regions) region-level connectivity.
    assignment
        Network label per region, length n_regions.
    networks
        Network names to use, in the order the columns should appear.

    Returns
    -------
    (within, between, within_names, between_names), where `within` is
    (n_subjects, len(networks)) and `between` is (n_subjects, n_pairs).

    Self-pairs are excluded from the within-network means, so a region's own
    diagonal entry -- constant for several statistics -- never enters.
    """
    matrices = np.asarray(matrices, dtype=float)
    assignment = np.asarray(assignment)
    if len(assignment) != matrices.shape[-1]:
        raise ValueError(f'{len(assignment)} assignments for '
                         f'{matrices.shape[-1]} regions')

    members = {net: np.flatnonzero(assignment == net) for net in networks}
    empty = [net for net, idx in members.items() if idx.size < 2]
    if empty:
        raise ValueError(f'networks with fewer than 2 regions: {empty}')

    within, within_names = [], []
    for net in networks:
        idx = members[net]
        block = matrices[:, idx[:, None], idx[None, :]]
        iu = np.triu_indices(len(idx), k=1)
        within.append(block[:, iu[0], iu[1]].mean(axis=1))
        within_names.append(f'{net}-{net}')

    between, between_names = [], []
    for i, a in enumerate(networks):
        for b in list(networks)[i + 1:]:
            block = matrices[:, members[a][:, None], members[b][None, :]]
            between.append(block.reshape(len(matrices), -1).mean(axis=1))
            between_names.append(f'{a}-{b}')

    return (np.stack(within, axis=1), np.stack(between, axis=1),
            within_names, between_names)


def partial_correlation(x, y, z) -> tuple[float, float]:
    """Correlation of x and y with z linearly removed from both.

    Returns (r, two-sided p). The p-value uses n - 3 degrees of freedom: one
    lost to the intercept and one to the covariate.

    Used for the motion confound check -- "does the group difference in this
    edge survive removing head motion" -- where the covariate is a per-subject
    motion summary.
    """
    x, y, z = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    if not len(x) == len(y) == len(z):
        raise ValueError(f'lengths differ: {len(x)}, {len(y)}, {len(z)}')
    if len(x) < 4:
        raise ValueError('need at least 4 observations')

    design = np.column_stack([np.ones_like(z), z])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r, _ = stats.pearsonr(residual_x, residual_y)

    n = len(x)
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    return float(r), float(2 * stats.t.sf(abs(t), df=n - 3))
