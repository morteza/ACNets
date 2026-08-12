"""Can prior knowledge substitute for the data this design does not have?

Everything tried so far estimates the discriminant direction from 32
participants in 21 dimensions, which section 11 shows is hopeless: the
bias-corrected Mahalanobis distance over those 21 dimensions is negative. An
inductive bias attacks that differently. Rather than estimating the direction
better, it *constrains* the space of directions using knowledge that is not in
the data, spending fewer degrees of freedom on the sample.

Four biases, ordered by how much they assume:

**discovered-contrast.** The discriminant is fixed at w = +1 on CON-SMN and
FPN-SMN, zero elsewhere, so **zero degrees of freedom are spent on direction**
and only a threshold is fitted inside each fold. Read the name literally: those
two edges are the ones section 3.1 of the manuscript found the classifiers
leaning on *in this dataset*. They were not declared in advance, and this script
originally claimed they were.

**network-restricted.** Weaker. Use only edges touching the three networks the
contrast names (FPN, CON, SMN), estimating freely within that subspace. The
constraint is about *where*, not about direction.

**laplacian-smooth.** Structural rather than substantive. Edges sharing a
network node should carry similar weights, since a difference localised to one
network expresses itself across all of that network's edges. Implemented as a
generalised ridge whose penalty is the Laplacian of the edge adjacency graph,
applied as a feature-space whitening before an ordinary discriminant.

**rank-one.** The group difference is assumed to be a single dominant mode in
the 6 x 6 network space rather than 21 independent edge effects. Implemented by
projecting the difference matrix onto its leading eigenvector, estimated inside
each training fold.

Each is compared against the unbiased baseline from section 12 (shrinkage LDA,
0.719) and against the reliability-weighted best from section 13 (0.750), with
the same LOOCV estimator and a family-wise null over the whole set.

**On circularity: this analysis is circular, and the accuracies below are not
p-values you may quote.** An earlier version of this docstring argued that the
direction came from prior literature rather than from this dataset's own
attribution ranking, "which happens to agree". It does not come from the
literature. Gong et al. (2015) report increased coupling between attentional and
sensorimotor systems in action-game experts, which motivates *looking* at
control-to-sensorimotor coupling; it does not name FPN, CON and SMN in the
Dosenbach vocabulary, and their "attentional" network is not the Dosenbach CON
or FPN. The specific triple was selected because section 3.1 found it. Selecting
features on all 32 participants and then scoring a classifier on the same 32 is
circular no matter how few degrees of freedom the final step spends.

The numbers are still worth computing, for two reasons. They quantify how much a
restriction of this kind is worth *at all*, which bounds what any honest
version could achieve. And they are the input to the comparison that does carry
information: the same restriction tested in an independent sample
(`run_crossmodal_networks.py`, which fails in both directions) and against
size-matched restrictions elsewhere in the brain
(`run_cerebellar_control.py`, which finds no advantage). Treat what follows as
an upper bound on a hypothesis, not as a test of one.

    uv run python scripts/run_inductive_bias.py --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.linalg import fractional_matrix_power  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse import accuracy_interval, max_stat_permutation  # noqa: E402
from acnets.multiverse.runner import edge_names  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402
from run_reliability_transfer import connectivity, loocv  # noqa: E402

OUT = Path('outputs/inductive_bias')
STATISTIC = 'precision'          # the best statistic from sections 12-13

# Discovered in *this* dataset (manuscript section 3.1), not declared in advance.
# Every accuracy computed from these constants is therefore an upper bound, not
# a test. Naming them for what they are is the only thing keeping the rest of
# this script from reading as a confirmatory analysis.
DISCOVERED_EDGES = ('CON ↔ SMN', 'FPN ↔ SMN')
DISCOVERED_NETWORKS = ('FPN', 'CON', 'SMN')


def edge_node_pairs(names: list[str]) -> list[tuple[str, str]]:
    return [tuple(n.split(' ↔ ')) for n in names]


def discovered_contrast(x: np.ndarray, y: np.ndarray, names: list[str]) -> float:
    """Fixed direction, fitted threshold. One degree of freedom, not 21.

    The direction is fixed, but it was fixed by looking at these participants,
    so the one degree of freedom is not the whole cost of the analysis.
    """
    columns = [i for i, n in enumerate(names) if n in DISCOVERED_EDGES]
    projection = x[:, columns].mean(axis=1)

    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        # the only thing learned is where to cut, and which side is which --
        # the direction itself is never estimated from these participants
        centres = [projection[train][y[train] == k].mean() for k in (0, 1)]
        threshold = float(np.mean(centres))
        higher = int(centres[1] > centres[0])
        above = projection[test][0] > threshold
        predictions[test] = higher if above else 1 - higher
    return float((predictions == y).mean())


def network_restricted(x: np.ndarray, names: list[str]) -> np.ndarray:
    pairs = edge_node_pairs(names)
    keep = [i for i, (a, b) in enumerate(pairs)
            if a in DISCOVERED_NETWORKS and b in DISCOVERED_NETWORKS]
    return x[:, keep]


def laplacian_whitening(names: list[str], strength: float) -> np.ndarray:
    """(I + strength * L)^(-1/2) for the edge-adjacency Laplacian.

    Two edges are adjacent when they share a network. Multiplying features by
    this matrix makes the subsequent ridge penalty behave like a smoothness
    penalty: weights that vary sharply between edges sharing a node are
    penalised more than weights that vary smoothly.
    """
    pairs = edge_node_pairs(names)
    n = len(pairs)
    adjacency = np.zeros((n, n))
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            if i != j and {a, b} & {c, d}:
                adjacency[i, j] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    return np.real(fractional_matrix_power(
        np.eye(n) + strength * laplacian, -0.5))


def rank_one(x: np.ndarray, y: np.ndarray, names: list[str]) -> float:
    """Project onto the leading mode of the group-difference matrix.

    The difference is assumed to be one dominant network-level mode rather than
    21 independent edge effects. The mode is estimated inside each training fold,
    so the assumption is the bias; the mode itself is not prior knowledge.
    """
    pairs = edge_node_pairs(names)
    nodes = sorted({n for pair in pairs for n in pair})
    index = {n: i for i, n in enumerate(nodes)}

    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        difference = (x[train][y[train] == 0].mean(axis=0)
                      - x[train][y[train] == 1].mean(axis=0))
        matrix = np.zeros((len(nodes), len(nodes)))
        for value, (a, b) in zip(difference, pairs, strict=True):
            matrix[index[a], index[b]] = value
            matrix[index[b], index[a]] = value
        _, vectors = np.linalg.eigh(matrix)
        leading = vectors[:, -1]
        direction = np.array([leading[index[a]] * leading[index[b]]
                              for a, b in pairs])

        scores = x @ direction
        centres = [scores[train][y[train] == k].mean() for k in (0, 1)]
        higher = int(centres[1] > centres[0])
        above = scores[test][0] > float(np.mean(centres))
        predictions[test] = higher if above else 1 - higher
    return float((predictions == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, args.seed)
    names = edge_names(nodes['network'], False)
    x = connectivity(timeseries['network'], STATISTIC)

    whitening = {s: laplacian_whitening(names, s) for s in (0.5, 2.0)}
    restricted = network_restricted(x, names)
    print(f'{len(y)} participants, statistic = {STATISTIC}')
    print(f'  full edge set          {x.shape[1]} features')
    print(f'  network-restricted     {restricted.shape[1]} features')
    print('  discovered contrast    1 fixed direction, 0 estimated\n')

    def evaluate(labels: np.ndarray) -> dict[str, float]:
        out = {
            'baseline/shrinkage-lda': loocv(x, labels),
            'discovered-contrast': discovered_contrast(x, labels, names),
            'network-restricted': loocv(network_restricted(x, names), labels),
            'rank-one': rank_one(x, labels, names),
        }
        for strength, matrix in whitening.items():
            out[f'laplacian-smooth/{strength}'] = loocv(x @ matrix, labels)
        return out

    observed = evaluate(y)
    table = pd.DataFrame([{'cell': k, 'accuracy': v}
                          for k, v in observed.items()]).sort_values(
        'accuracy', ascending=False)
    print(f'{"bias":34s} {"accuracy":>9s}   Wilson 95%')
    for _, row in table.iterrows():
        interval = accuracy_interval(row['accuracy'] * len(y), len(y))
        print(f'  {row["cell"]:32s} {row["accuracy"]:9.3f}   '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

    started = time.perf_counter()
    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}:')

    def one(i: int) -> list[float]:
        rng = np.random.default_rng(args.seed + 1 + i)
        scores = evaluate(rng.permutation(y))
        return [scores[k] for k in observed]

    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(one)(i) for i in range(args.n_permutations)))
    print(f'  {time.perf_counter() - started:.0f}s')

    result = max_stat_permutation(observed, null)
    print(f'\n  observed max        {result.observed_max:.3f}')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  null max, 95th pct  {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')

    # Reported to show the size of the gap between a corrected and an
    # uncorrected reading of the same number -- NOT as a licensed test. An
    # uncorrected p is only meaningful for a hypothesis fixed before the data
    # were seen, and this one was not (see the docstring).
    column = list(observed).index('discovered-contrast')
    alone = null[:, column]
    p_alone = ((1 + (alone >= observed['discovered-contrast']).sum())
               / (len(alone) + 1))
    print('\n  discovered contrast, scored against its own null alone:')
    print(f'    accuracy {observed["discovered-contrast"]:.3f}, '
          f'null median {np.median(alone):.3f}, p = {p_alone:.4f}')
    print('    ^ NOT a licensed p-value: the edges were chosen from these same')
    print('      32 participants (manuscript section 3.1). Quoted only to show')
    print('      what the circularity is worth.')

    (OUT / 'inductive_bias.json').write_text(json.dumps({
        'statistic': STATISTIC,
        'discovered_edges': list(DISCOVERED_EDGES),
        'provenance': ('the edges and networks used here were selected from '
                       'this dataset (manuscript 3.1), not declared in '
                       'advance; every accuracy is an upper bound, and '
                       'discovered_contrast_uncorrected_p is reported only to '
                       'quantify the circularity, not as a test'),
        'observed': observed, **result.summary(),
        'discovered_contrast_uncorrected_p': float(p_alone),
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
