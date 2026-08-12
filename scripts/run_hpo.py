"""Does a better classifier, properly tuned, decode the groups?

Two questions at once, kept apart because they have different standing.

**Does tuning change anything?** Methods 6.5 claimed hyperparameters were tuned
on training data; the code used fixed values (`acnets.multiverse.classifiers`).
Running both settings makes the claim true and measures what it was worth. This
is a correction, not a search: nested tuning does not inflate an estimate.

**Does a wider set of model families help?** Thirteen families instead of three,
biased toward the heavily regularised linear models that suit n = 32, plus `qda`
and `knn` which should cope badly and so give the comparison a floor above
chance. This *is* a search, it widens the family-wise null, and it is reported
under that null rather than as a best-of.

Both run over several feature sets, because a classifier comparison on one
feature set says little: the raw 21 edges, the two attributed edges, and the
normative deviation scores if `run_normative.py` has produced them.

Parallelism is over (feature set x classifier x tuning) cells with joblib, and
the inner hyperparameter search is kept single-threaded so the two levels do not
oversubscribe the cores. The permutation null reuses the same parallel map.

    uv run python scripts/run_hpo.py
    uv run python scripts/run_hpo.py --n-permutations 0 --classifiers ridge lda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# xgboost before torch; see acnets._openmp
from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure  # noqa: E402
from acnets.multiverse import (  # noqa: E402
    accuracy_interval,
    fixed_splits,
    max_stat_permutation,
)
from acnets.multiverse.classifiers import available, make_classifier  # noqa: E402
from acnets.multiverse.runner import edge_names, vectorize  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402

OUT = Path('outputs/hpo')
STATISTIC = 'partial correlation'


def feature_sets(timeseries, nodes) -> dict[str, np.ndarray]:
    measure = ExtraConnectivityMeasure(kind=STATISTIC, vectorize=False)
    measure.fit(timeseries['network'])
    raw = vectorize(measure.transform(timeseries['network']), False)
    names = edge_names(nodes['network'], False)

    sets = {'raw-21': raw}

    top = [i for i, n in enumerate(names) if n in ('CON ↔ SMN', 'FPN ↔ SMN')]
    if len(top) == 2:
        # circular by construction -- these two edges were chosen on these data.
        # Included so the classifier comparison has a case with real signal in
        # it, and labelled so nobody reads it as an out-of-sample estimate.
        sets['top2-circular'] = raw[:, top]

    scores = Path('outputs/normative/scores.npz')
    if scores.exists():
        sets['normative-diffusion'] = np.load(scores)['diffusion']

    return sets


def score(x, y, splits, name, tuned, n_iter, seed) -> tuple[float, str]:
    """Mean outer-fold accuracy, with any tuning done inside the training fold.

    Returns (accuracy, error). A family that cannot fit this shape of data --
    QDA needs more samples per class than features, and a training fold here has
    12 and 21 -- returns NaN and the reason rather than killing the sweep. That
    failure is a fact about the regime worth reporting, not an accident.
    """
    folds = []
    for train, test in splits:
        scaler = StandardScaler().fit(x[train])
        model = make_classifier(name, random_state=seed, tuned=tuned,
                                n_iter=n_iter, n_jobs=1)
        try:
            model.fit(scaler.transform(x[train]), y[train])
            predicted = model.predict(scaler.transform(x[test]))
        except Exception as error:
            return float('nan'), f'{type(error).__name__}: {str(error)[:120]}'
        folds.append(float((predicted == y[test]).mean()))
    return float(np.mean(folds)), ''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifiers', nargs='*', default=None)
    parser.add_argument('--n-iter', type=int, default=20,
                        help='hyperparameter configurations sampled per fold')
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=25)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, args.seed)
    sets = feature_sets(timeseries, nodes)
    names = args.classifiers or available()

    cells = [(family, classifier, tuned)
             for family in sets
             for classifier in names
             for tuned in (False, True)]
    splits = fixed_splits(len(y), args.n_splits, args.n_repeats, args.seed)

    print(f'{len(sets)} feature sets x {len(names)} classifiers x 2 tuning '
          f'settings = {len(cells)} specifications')
    print('features: ' + ', '.join(f'{k} ({v.shape[1]})' for k, v in sets.items()))
    print(f'classifiers: {", ".join(names)}')
    print(f'CV {args.n_splits} x {args.n_repeats}, '
          f'inner search {args.n_iter} configurations per fold when tuned\n')

    started = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(score)(sets[family], y, splits, classifier, tuned,
                       args.n_iter, args.seed)
        for family, classifier, tuned in cells)
    print(f'observed grid in {time.perf_counter() - started:.0f}s\n')

    rows = []
    for (family, classifier, tuned), (value, error) in zip(cells, results,
                                                           strict=True):
        interval = (accuracy_interval(value * len(y), len(y))
                    if np.isfinite(value) else None)
        rows.append({'features': family, 'classifier': classifier,
                     'tuned': tuned, 'accuracy': value,
                     'wilson_low': interval.low if interval else float('nan'),
                     'wilson_high': interval.high if interval else float('nan'),
                     'error': error})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'hpo.csv', index=False)

    failures = table[table['error'] != '']
    for _, row in failures.iterrows():
        print(f'  could not fit: {row["features"]}/{row["classifier"]}'
              f'{"+tuned" if row["tuned"] else ""} -- {row["error"]}')
    if len(failures):
        print()
    # a cell that cannot fit is dropped from the null rather than scored as
    # chance, which would quietly drag the maximum down
    cells = [c for c, (v, _) in zip(cells, results, strict=True)
             if np.isfinite(v)]
    observed = {f'{family}/{classifier}{"+tuned" if tuned else ""}': value
                for (family, classifier, tuned), (value, _) in
                zip(cells, [r for r in results if np.isfinite(r[0])], strict=True)}

    for family in sets:
        block = table[table['features'] == family]
        print(f'{family}:')
        pivot = block.pivot_table(index='classifier', columns='tuned',
                                  values='accuracy')
        pivot.columns = ['untuned', 'tuned']
        pivot['delta'] = pivot['tuned'] - pivot['untuned']
        for classifier, row in pivot.sort_values('tuned', ascending=False).iterrows():
            print(f'  {classifier:16s} untuned {row["untuned"]:.3f}   '
                  f'tuned {row["tuned"]:.3f}   {row["delta"]:+.3f}')
        print()

    gain = table.pivot_table(index=['features', 'classifier'], columns='tuned',
                             values='accuracy')
    gain.columns = ['untuned', 'tuned']
    delta = (gain['tuned'] - gain['untuned'])
    print(f'effect of tuning across all {len(delta)} cells: '
          f'mean {delta.mean():+.3f}, median {delta.median():+.3f}, '
          f'best {delta.max():+.3f}, worst {delta.min():+.3f}')
    print(f'tuning helped in {int((delta > 0).sum())} of {len(delta)} cells\n')

    payload = {'n_subjects': len(y), 'n_specifications': len(cells),
               'cv': {'n_splits': args.n_splits, 'n_repeats': args.n_repeats},
               'n_iter': args.n_iter,
               'feature_sets': {k: int(v.shape[1]) for k, v in sets.items()},
               'classifiers': list(names),
               'tuning_effect': {'mean': float(delta.mean()),
                                 'median': float(delta.median()),
                                 'max': float(delta.max()),
                                 'min': float(delta.min()),
                                 'n_improved': int((delta > 0).sum()),
                                 'n_cells': len(delta)},
               'observed': observed}

    if args.n_permutations:
        print(f'family-wise null over all {len(cells)} specifications, '
              f'B = {args.n_permutations}:')
        rng = np.random.default_rng(args.seed)
        null = np.empty((args.n_permutations, len(observed)))
        for i in range(args.n_permutations):
            permuted = rng.permutation(y)
            values = Parallel(n_jobs=args.n_jobs)(
                delayed(score)(sets[family], permuted, splits, classifier, tuned,
                               args.n_iter, args.seed)
                for family, classifier, tuned in cells)
            null[i] = [v for v, _ in values]
            if (i + 1) % 20 == 0:
                rate = (time.perf_counter() - started) / (i + 1)
                print(f'  {i + 1}/{args.n_permutations} '
                      f'(~{rate * (args.n_permutations - i - 1) / 60:.0f} min left)',
                      flush=True)

        result = max_stat_permutation(observed, null)
        best = max(observed, key=observed.get)
        print(f'\n  best                {observed[best]:.3f}  ({best})')
        print(f'  null max, median    {np.median(result.null_max):.3f}')
        print(f'  critical value (95) {result.critical_value:.3f}')
        print(f'  p                   {result.p_value:.4f}')
        n_significant = sum(v >= result.critical_value for v in observed.values())
        print(f'  {n_significant} of {len(observed)} clear it')
        payload.update(result.summary())
        payload['n_significant'] = int(n_significant)

    (OUT / 'hpo.json').write_text(json.dumps(payload, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
