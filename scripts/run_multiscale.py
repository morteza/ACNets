"""Section 3.2: do regional, within-network or between-network features carry the signal?

The three accounts of breadth in section 2.1 make claims about different parts
of the same connectivity structure, and comparing them across separate models is
unfair because the models differ in dimensionality. Putting all three blocks in
one classifier lets them compete on equal terms.

**Why this does not use `acnets.pipeline.MultiScaleClassifier`.** That class
defines its regional block as `dataset['timeseries'].mean('timepoint')`. The
parcellated time courses are z-scored, so those means are zero up to numerical
residue -- measured here at max |mean| = 0.033 against a per-region SD of 0.996.
Its regional block is therefore noise by construction, and a finding that
"between-network features beat regional ones" computed that way would be
guaranteed rather than discovered. The per-region SD is no better: it spans
0.93 to 1.00 by the same standardisation.

The blocks used instead, all derived from one connectivity matrix so that no
block gets an estimation advantage:

  H1  regional  weighted degree of each region, i.e. its mean absolute
                connectivity to every other region. One feature per region,
                restricted to the three networks of interest (86 features).
                A real regional summary that standardisation cannot flatten.
  H2  within    mean connectivity among the region pairs inside each network,
                one per network (6).
  H3  between   mean connectivity among the region pairs spanning each network
                pair, one per pair (15).

All three come from the *same* region-level matrix. An earlier version took H2
and H3 from the network-aggregated matrix's diagonal and upper triangle, which
is wrong for the statistics whose diagonal is fixed: a partial-correlation
matrix has a constant diagonal, so H2 was a constant block. It showed up as all
three classifiers scoring an identical 0.370 on H2 alone and losing exactly
0.000 when H2 was shuffled. Deriving both blocks from region pairs avoids the
problem for every statistic and matches how "mean within-network connectivity"
is normally defined anyway.

Block importance is measured two ways, because they answer different questions
and can disagree:

  *Ablation.* Fit on each block alone and on all three. What a block can do by
  itself, which is the quantity the three accounts actually make claims about.
  *Permutation.* Fit on all three, then shuffle one block's columns within the
  test fold and measure the accuracy drop. What a block contributes *given* the
  others, which is what a single fitted model relies on.

Everything is fitted inside training folds on the same label-independent splits
used throughout, and reported with the Wilson interval at n = 32 rather than the
fold spread.

    uv run python scripts/run_multiscale.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.analysis.multiscale import network_blocks, regional_degree
from acnets.connectome import ExtraConnectivityMeasure
from acnets.multiverse import accuracy_interval, fixed_splits
from acnets.multiverse.runner import _make_classifier
from run_agreement import aggregate, group_labels, load_atlas

OUT = Path('outputs/multiscale')

STATISTIC = 'partial correlation'   # the primary specification's statistic
CLASSIFIERS = ('svm', 'xgboost', 'mlp')
NETWORKS_OF_INTEREST = ('FPN', 'CON', 'SMN')


def connectivity(timeseries: np.ndarray) -> np.ndarray:
    measure = ExtraConnectivityMeasure(kind=STATISTIC, vectorize=False)
    measure.fit(timeseries)
    return measure.transform(timeseries)


def evaluate(x, y, splits, name, seed=0, shuffle_block=None, rng_seed=0):
    """Mean accuracy over folds; optionally shuffle a column block in the test fold."""
    rng = np.random.default_rng(rng_seed)
    folds = []
    for train, test in splits:
        scaler = StandardScaler().fit(x[train])
        x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
        if shuffle_block is not None:
            x_test = x_test.copy()
            columns = x_test[:, shuffle_block]
            x_test[:, shuffle_block] = columns[rng.permutation(len(x_test))]
        model = _make_classifier(name, seed)
        model.fit(x_train, y[train])
        folds.append(float((model.predict(x_test) == y[test]).mean()))
    return float(np.mean(folds))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, 0)

    # H1 is computed on region-level connectivity, then restricted to the three
    # networks of interest; the remaining regions are dropped rather than
    # summarised, so the block stays comparable in size to the others.
    network_names = nodes['network']
    # region-to-network assignment lives in a dataset variable aligned to the
    # region axis; region *names* do not encode it
    region_networks = [str(n) for n in np.asarray(dataset['network'].values)]
    keep = np.array([i for i, net in enumerate(region_networks)
                     if net in NETWORKS_OF_INTEREST])
    if keep.size == 0:
        raise SystemExit(f'no regions matched {NETWORKS_OF_INTEREST}; '
                         f'assignments seen: {sorted(set(region_networks))}')

    region_matrices = connectivity(timeseries['region'])
    h1 = regional_degree(region_matrices)[:, keep]
    h2, h3, h2_names, h3_names = network_blocks(
        region_matrices, region_networks, list(network_names))

    blocks = {'H1 regional': h1, 'H2 within': h2, 'H3 between': h3}
    print(f'{len(y)} participants; blocks: '
          + ', '.join(f'{k} {v.shape[1]}' for k, v in blocks.items()))
    print(f'networks: {network_names}')
    for name, block in (('H2', h2), ('H3', h3)):
        constant = int((np.ptp(block, axis=0) < 1e-12).sum())
        if constant:
            raise SystemExit(f'{name} has {constant} constant columns')
    print('no constant columns in H2 or H3\n')

    splits = fixed_splits(len(y), n_splits=4, n_repeats=25, random_state=0)
    combined = np.hstack(list(blocks.values()))
    spans, start = {}, 0
    for name, block in blocks.items():
        spans[name] = np.arange(start, start + block.shape[1])
        start += block.shape[1]

    payload = {'statistic': STATISTIC, 'n_subjects': len(y),
               'block_sizes': {k: int(v.shape[1]) for k, v in blocks.items()},
               'networks_of_interest': list(NETWORKS_OF_INTEREST),
               'h2_features': h2_names, 'h3_features': h3_names,
               'alone': {}, 'combined': {}, 'permutation_drop': {}}

    # The rest of the paper aggregates *timeseries* into networks and then
    # estimates connectivity; the blocks above estimate region-level
    # connectivity and then average it. The two are not the same operation and
    # the difference is worth measuring, so the network-scale off-diagonal --
    # the same 15 quantities H3 describes, computed the other way round -- is
    # evaluated alongside as a reference rather than as a fourth block.
    network_matrices = connectivity(timeseries['network'])
    iu = np.triu_indices(network_matrices.shape[-1], k=1)
    h3_aggregated = network_matrices[:, iu[0], iu[1]]
    payload['h3_aggregate_then_connect'] = {}

    for classifier in CLASSIFIERS:
        print(f'{classifier}:')
        for name, block in blocks.items():
            accuracy = evaluate(block, y, splits, classifier)
            interval = accuracy_interval(accuracy * len(y), len(y))
            payload['alone'].setdefault(classifier, {})[name] = {
                'accuracy': accuracy,
                'wilson_ci': [interval.low, interval.high]}
            print(f'  {name:14s} alone  {accuracy:.3f}  '
                  f'[{interval.low:.2f}, {interval.high:.2f}]')

        full = evaluate(combined, y, splits, classifier)
        interval = accuracy_interval(full * len(y), len(y))
        payload['combined'][classifier] = {
            'accuracy': full, 'wilson_ci': [interval.low, interval.high]}
        print(f'  {"all three":14s}        {full:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

        for name in blocks:
            shuffled = evaluate(combined, y, splits, classifier,
                                shuffle_block=spans[name])
            drop = full - shuffled
            payload['permutation_drop'].setdefault(classifier, {})[name] = drop
            print(f'  {name:14s} drop   {drop:+.3f}')

        reference = evaluate(h3_aggregated, y, splits, classifier)
        interval = accuracy_interval(reference * len(y), len(y))
        payload['h3_aggregate_then_connect'][classifier] = {
            'accuracy': reference, 'wilson_ci': [interval.low, interval.high]}
        print(f'  {"H3, aggregate-then-connect":26s} {reference:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]')
        print()

    (OUT / 'multiscale.json').write_text(json.dumps(payload, indent=2, default=float))
    print(f'written to {OUT}/')


if __name__ == '__main__':
    main()
