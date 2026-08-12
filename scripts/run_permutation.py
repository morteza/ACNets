"""Family-wise permutation null over the whole specification grid.

The question this answers is the one a multiverse creates for itself: having
tried 24 analysis paths and reported the best, how surprising is that best?

`acnets.multiverse.inference` calls the absence of this test "the old paper's
fatal flaw", implements `max_stat_permutation` to fix it, and -- until this
script -- nothing ever called it. Every accuracy in `outputs/agreement/` was
therefore uncorrected for having searched the space.

Method. Labels are permuted *outside everything*: outside the connectivity
statistic, outside scaling, outside the classifier. For each permutation the
entire grid is re-run and the maximum accuracy across specifications is kept.
The observed maximum is then compared against that distribution, which is exact
rather than conservative like Bonferroni, and which automatically accounts for
the correlation between specifications fitted to the same 32 people.

Feature extraction is unsupervised -- correlation, tangent and the rest never
see `y` -- so permuting labels while reusing the aggregated timeseries is
legitimate and is what makes this affordable. Attribution is skipped, since the
null is about accuracy only, and that is most of the cost for the tree models.

Both the observed and the null runs use the same cross-validation budget. They
have to: a mean over 100 folds is less variable than a mean over 20, and
comparing a smooth observed maximum against a noisy null maximum would inflate
significance.

    uv run python scripts/run_permutation.py --n-permutations 200
    uv run python scripts/run_permutation.py --n-permutations 20 --n-repeats 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# so `run_agreement` resolves whether or not scripts/ is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse import (
    SpecificationRegistry,
    accuracy_interval,
    fixed_splits,
    max_stat_permutation,
    run_specifications,
    specification_curve,
)
from run_agreement import (
    CACHE,
    RUNNABLE_STATISTICS,
    aggregate,
    group_labels,
    lemon_transforms,
    load_atlas,
)

OUT = Path('outputs/permutation')


def run_grid(specs, timeseries, nodes, labels, splits, seed, n_jobs, lemon=None):
    """Accuracy per specification, no attribution.

    `splits` is one label-independent partition (`fixed_splits`) shared by the
    observed run and every permutation. That symmetry is a validity requirement
    -- otherwise the observed statistic is computed on label-stratified folds
    while the null is not -- and it is also what lets the feature cache hit
    across permutations: features never see the labels, so with the folds held
    fixed every permutation reuses the same cached matrices and pays only for
    classifier refits.
    """
    outcome = run_specifications(
        specs, timeseries, nodes, labels,
        random_state=seed, n_jobs=n_jobs,
        compute_attribution=False,
        lemon=lemon, cache_dir=CACHE, splits=splits)
    return {r.spec.label: r.accuracy
            for r in outcome.results.values() if r.failed is None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='CV repeats, used identically for observed and null')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--no-lemon', action='store_true',
                        help='drop the lemon-pretrained specifications')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    dataset = load_atlas('dosenbach2010')
    labels, _ = group_labels(dataset)
    n_subjects = len(labels)
    timeseries, nodes = aggregate(dataset, args.seed)

    registry = SpecificationRegistry(
        atlases=('dosenbach2010',),
        statistics=RUNNABLE_STATISTICS,
        aggregations=('network', 'random_network'),
        classifiers=('svm', 'xgboost', 'mlp'),
        pretrainings=('none',) if args.no_lemon else ('none', 'lemon'))
    specs = registry.specifications

    lemon = None
    if any(s.pretraining == 'lemon' for s in specs):
        lemon = lemon_transforms('dosenbach2010', nodes, RUNNABLE_STATISTICS,
                                 window_length=dataset.sizes['timepoint'])
        if lemon is None:
            specs = [s for s in specs if s.pretraining == 'none']

    print(f'{len(specs)} specifications, {n_subjects} subjects, '
          f'{args.n_splits} x {args.n_repeats} CV\n')

    splits = fixed_splits(n_subjects, args.n_splits, args.n_repeats, args.seed)

    observed = run_grid(specs, timeseries, nodes, labels, splits,
                        args.seed, args.n_jobs, lemon=lemon)
    best = max(observed, key=observed.get)
    print(f'observed best: {observed[best]:.3f}  {best}')
    print(f'observed grid: {min(observed.values()):.3f} to '
          f'{max(observed.values()):.3f}\n')

    rng = np.random.default_rng(args.seed)
    null = np.empty((args.n_permutations, len(observed)), dtype=float)
    started = time.perf_counter()

    for i in range(args.n_permutations):
        # permuted outside the whole pipeline; features never saw y anyway
        permuted = rng.permutation(labels)
        scores = run_grid(specs, timeseries, nodes, permuted, splits,
                          args.seed, args.n_jobs, lemon=lemon)
        null[i] = [scores[k] for k in observed]

        if (i + 1) % 10 == 0:
            rate = (time.perf_counter() - started) / (i + 1)
            print(f'  {i + 1}/{args.n_permutations} permutations  '
                  f'({rate:.1f}s each, ~{rate * (args.n_permutations - i - 1) / 60:.1f} '
                  f'min left)', flush=True)

    result = max_stat_permutation(observed, null)

    print(f'\nfamily-wise null over {result.n_specifications} specifications, '
          f'{result.n_permutations} permutations')
    print(f'  observed max        {result.observed_max:.3f}')
    print(f'  null max, mean      {null.max(axis=1).mean():.3f}')
    print(f'  critical value (95) {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}'
          f'   (floor {1 / (args.n_permutations + 1):.4f})')

    wilson = accuracy_interval(result.observed_max * n_subjects, n_subjects)
    print(f'  best spec sampling CI {wilson}')

    curve = specification_curve(observed, null_max=result.null_max)
    frame = pd.DataFrame(curve)
    frame.to_csv(OUT / 'specification_curve.csv', index=False)

    n_sig = int(frame['significant'].sum())
    print(f'\n{n_sig} of {len(frame)} specifications clear the family-wise '
          f'critical value')
    if n_sig:
        for _, r in frame[frame['significant']].iterrows():
            print(f'  {r["accuracy"]:.3f}  {r["spec_id"]}')

    (OUT / 'permutation.json').write_text(json.dumps({
        'n_subjects': n_subjects,
        'n_splits': args.n_splits,
        'n_repeats': args.n_repeats,
        'observed': observed,
        **result.summary(),
        'null_max_quantiles': {
            str(q): float(np.quantile(result.null_max, q))
            for q in (0.5, 0.9, 0.95, 0.99)},
    }, indent=2, default=float))
    np.save(OUT / 'null_max.npy', result.null_max)
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
