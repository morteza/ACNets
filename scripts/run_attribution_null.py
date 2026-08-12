"""A permuted-label null for the cross-specification attribution agreement.

The agreement analysis in `run_agreement.py` calibrates observed attribution
agreement against *random rankings of the same feature vocabulary*. That null
answers one question -- is the agreement bigger than the floor a shared
vocabulary manufactures -- and leaves a second one open, which is the one a
sceptical reader asks first:

    the classifiers do not decode above a corrected null, so could their
    attributions agree for reasons that have nothing to do with the group
    labels?

They could. Haufe's forward pattern is A = Sigma w / (w' Sigma w). As the
decoder weight w approaches noise, A approaches Sigma w, whose ranking is
governed by the feature covariance -- a label-free quantity that every
specification shares. Tree and gradient attributions have their own versions of
the same failure. If that is what is happening, then refitting the whole grid on
permuted labels should reproduce the observed agreement, because nothing that
produced it depended on the labels being real.

So this script destroys the labels and leaves everything else alone: same
specifications, same label-independent folds, same cached features, same
attribution methods. For each permutation it recomputes the full pairwise
agreement matrix over the grid and keeps its mean. The result is the
distribution of attribution agreement under the null hypothesis that group
membership carries no information -- which is the null the claim actually needs.

Two things make this cheap enough to run at a useful B. Features never see the
labels, so the cache written by `run_agreement.py` is valid for every
permutation and only the classifier refits are paid for. And `--classifiers svm`
restricts the grid to the linear specifications, whose Haufe attribution is a
closed-form covariance product rather than a per-fold SHAP pass; that is also
the channel with the sharpest theoretical rival, since it is the one whose
degenerate case is literally Sigma w.

Usage:
    uv run python scripts/run_attribution_null.py                    # B = 200, linear
    uv run python scripts/run_attribution_null.py -B 50 --all-classifiers
    uv run python scripts/run_attribution_null.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from acnets.multiverse import (
    Specification,
    agreement_matrix,
    fixed_splits,
    random_agreement_baseline,
    run_specifications,
)
from acnets.multiverse.runner import collapse_to_undirected
from acnets.pipeline import TimeseriesAggregator

DATA = Path('data/Julia2018')
CACHE = Path('outputs/cache/features')
OUT = Path('outputs/agreement')

STATISTICS = ('correlation', 'partial correlation', 'covariance', 'precision',
              'tangent', 'chatterjee')


def build_grid(classifiers: tuple[str, ...]) -> list[Specification]:
    """Network-scale specifications on the one atlas the agreement analysis uses."""
    return [Specification(atlas='dosenbach2010', statistic=statistic,
                          aggregation='network', classifier=classifier,
                          pretraining='none')
            for statistic in STATISTICS
            for classifier in classifiers]


def mean_agreement(attributions: dict[str, dict[str, float]], seed: int) -> float:
    """Mean pairwise RBO over a grid, in the same units as the observed value."""
    collapsed = {k: collapse_to_undirected(v) for k, v in attributions.items()}
    return agreement_matrix(collapsed, method='rbo', random_state=seed).mean


def consensus_leader(attributions: dict[str, dict[str, float]]) -> str:
    """The feature the grid puts first on average, by mean rank across specifications.

    Agreement and consensus are different questions and the null answers both.
    A permutation can produce specifications that agree strongly with each other
    -- they overfit the same 32 participants along the same folds -- while
    landing on a different leading feature every time. Recording which feature
    leads therefore tests the substantive claim of section 3.1 (that CON-SMN is
    what the multiverse converges on) rather than the statistical one.
    """
    ranks: dict[str, list[int]] = {}
    for attribution in attributions.values():
        collapsed = collapse_to_undirected(attribution)
        order = sorted(collapsed, key=lambda f: -abs(collapsed[f]))
        for position, feature in enumerate(order):
            ranks.setdefault(feature, []).append(position)
    return min(ranks, key=lambda f: float(np.mean(ranks[f])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-B', '--n-permutations', type=int, default=200)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=25)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='',
                        help='suffix for the output filenames, so concurrent '
                             'runs of different grids do not overwrite each other')
    parser.add_argument('--all-classifiers', action='store_true',
                        help='add xgboost and mlp; several times slower because '
                             'SHAP is recomputed per fold per permutation')
    args = parser.parse_args()

    classifiers = ('svm', 'xgboost', 'mlp') if args.all_classifiers else ('svm',)
    specs = build_grid(classifiers)

    dataset = xr.open_dataset(DATA / 'timeseries_dosenbach2010.nc5',
                              engine='h5netcdf').load()
    np.random.seed(args.seed)
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    timeseries = aggregated['timeseries'].transpose(
        'subject', 'timepoint', 'network').values
    nodes = [str(n) for n in aggregated['network'].values]

    subjects = [str(s) for s in dataset['subject'].values]
    labels = np.array([0 if s.startswith('AVGP') else 1 for s in subjects])
    n_subjects = labels.size

    splits = fixed_splits(n_subjects, args.n_splits, args.n_repeats,
                          random_state=args.seed)
    common = dict(node_names_by_aggregation={'network': nodes},
                  timeseries_by_aggregation={'network': timeseries},
                  splits=splits, n_jobs=args.n_jobs, cache_dir=CACHE,
                  compute_attribution=True)

    print(f'{len(specs)} specifications ({", ".join(classifiers)}), '
          f'{n_subjects} subjects, {len(splits)} folds, '
          f'B = {args.n_permutations}')

    observed_run = run_specifications(specs, labels=labels, **common)
    observed = mean_agreement(observed_run.labelled_attributions(), args.seed)
    observed_leader = consensus_leader(observed_run.labelled_attributions())
    accuracies = observed_run.labelled_accuracies()
    print(f'observed mean pairwise RBO: {observed:.4f}  '
          f'(accuracy {min(accuracies.values()):.3f}-{max(accuracies.values()):.3f})')
    print(f'observed consensus leader: {observed_leader}')

    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_permutations, dtype=float)
    leaders: list[str] = []
    started = time.time()
    for i in range(args.n_permutations):
        permuted = rng.permutation(labels)
        run = run_specifications(specs, labels=permuted, **common)
        null[i] = mean_agreement(run.labelled_attributions(), args.seed)
        leaders.append(consensus_leader(run.labelled_attributions()))
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - started) / (i + 1)
            print(f'  {i + 1}/{args.n_permutations}  '
                  f'null mean so far {null[:i + 1].mean():.4f}  '
                  f'({rate:.1f}s each, ~{rate * (args.n_permutations - i - 1) / 60:.1f} '
                  f'min left)')

    # Upper-tail p, bounded below by 1/(B+1) as in `max_stat_permutation`.
    p_value = float((1.0 + np.sum(null >= observed)) / (args.n_permutations + 1.0))

    # The random-ranking floor the paper already reports, for the same grid, so
    # the two nulls can be read side by side.
    vocabulary = sorted({f for a in observed_run.labelled_attributions().values()
                         for f in collapse_to_undirected(a)})
    ranking_floor = random_agreement_baseline(vocabulary, method='rbo',
                                              random_state=args.seed)

    leader_counts = {name: leaders.count(name) for name in sorted(set(leaders))}
    leader_p = float((1.0 + leaders.count(observed_leader))
                     / (args.n_permutations + 1.0))

    report = {
        'n_specifications': len(specs),
        'classifiers': list(classifiers),
        'n_subjects': n_subjects,
        'n_permutations': args.n_permutations,
        'observed_rbo': observed,
        'observed_leader': observed_leader,
        'leader_under_permutation': leader_counts,
        'leader_p_value': leader_p,
        'permuted_label_null': {
            'mean': float(null.mean()),
            'sd': float(null.std(ddof=1)),
            'q95': float(np.percentile(null, 95)),
            'max': float(null.max()),
        },
        'random_ranking_null': {
            'mean': float(ranking_floor.mean()),
            'q95': float(np.percentile(ranking_floor, 95)),
        },
        'excess_over_permuted_labels': observed - float(null.mean()),
        'p_value': p_value,
        'p_floor': 1.0 / (args.n_permutations + 1.0),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f'_{args.tag}' if args.tag else ''
    (OUT / f'attribution_null{suffix}.json').write_text(json.dumps(report, indent=2))
    np.save(OUT / f'attribution_null{suffix}.npy', null)

    print(f'\nobserved                  {observed:.4f}')
    print(f'permuted-label null       {null.mean():.4f} '
          f'(sd {null.std(ddof=1):.4f}, 95th {np.percentile(null, 95):.4f})')
    print(f'random-ranking floor      {ranking_floor.mean():.4f}')
    print(f'excess over permuted      {observed - null.mean():+.4f}')
    print(f'p                         {p_value:.4f} '
          f'(floor {1 / (args.n_permutations + 1):.4f})')
    print(f'\nconsensus leader          {observed_leader}')
    print(f'  led under permutation   {leader_counts.get(observed_leader, 0)}'
          f'/{args.n_permutations}  (p = {leader_p:.4f})')
    top_alternatives = sorted(leader_counts.items(), key=lambda kv: -kv[1])[:5]
    print('  most frequent leaders under permutation: '
          + ', '.join(f'{name} {count}' for name, count in top_alternatives))
    print(f'\nwrote {OUT / f"attribution_null{suffix}.json"}')


if __name__ == '__main__':
    main()
