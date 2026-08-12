"""Exploratory search for a feature family that actually separates the groups.

**This is a search, and its output is not a result.** Every accuracy printed
here is the maximum over things we tried, on the same 32 participants that
produced every other number in this project, so it carries no inferential weight
until it is put through the family-wise null in `run_permutation.py`. It exists
to answer a prior question -- *is there anything here worth adding to the grid?*
-- not to produce a headline.

The reason for asking is that the six connectivity statistics currently in the
multiverse are near-redundant: all six are functions of the same 6x6 covariance
of the same aggregated time courses. Adding a seventh view of one covariance
matrix cannot find signal the first six missed; it only widens the null. So this
script tests feature families that carry *different information*:

  static     the existing baseline, for reference
  dynamic    sliding-window connectivity -- the mean and the standard deviation
             of coupling over time. The SD is the new part: static connectivity
             discards it entirely, and time-varying coupling is a different
             property of the same run.
  strength   weighted node strength per region, split into positive and negative
             coupling, then averaged within network. A graph property rather
             than an edge property.
  region     the full region-level edge set with in-fold dimensionality
             reduction. The paper dismisses region scale on dimensionality
             grounds, but never tried reducing it *inside* the folds, which is
             the standard remedy and is not leakage.

and it runs all of them across all nine cached atlases rather than the one the
paper used, since spatial resolution is the axis most likely to carry real
signal that a 6-network summary destroys.

Correlation is used throughout because it is computed per participant with no
cross-participant fitting, so it can be computed once outside the folds without
leaking. Everything that *is* fitted -- scaling, reduction, the classifier --
is fitted inside training folds only, on the same label-independent splits used
everywhere else.

    uv run python scripts/explore_features.py
    uv run python scripts/explore_features.py --atlases dosenbach2010 gordon2014
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from acnets.multiverse import accuracy_interval, fixed_splits
from acnets.multiverse.runner import _make_classifier
from acnets.pipeline import TimeseriesAggregator

CACHE = Path('data/Julia2018/derivatives/resting_timeseries')
OUT = Path('outputs/explore')

CLASSIFIERS = ('svm', 'xgboost', 'mlp')
WINDOW, STRIDE = 30, 5          # sliding window in TRs (TR = 3 s, so 90 s)

# region-level families are skipped above these parcel counts; see families_for
REGION_CAP, DYNAMIC_CAP = 400, 200


def load(atlas: str) -> xr.Dataset:
    return xr.open_dataset(CACHE / f'timeseries_{atlas}.nc5')


def labels_of(dataset: xr.Dataset) -> np.ndarray:
    return np.array([0 if str(s).startswith('AVGP') else 1
                     for s in dataset['subject'].values])


def correlations(timeseries: np.ndarray) -> np.ndarray:
    """Per-subject correlation matrices. No cross-subject fitting, so leak-free."""
    out = []
    for run in timeseries:
        centred = run - run.mean(axis=0, keepdims=True)
        sd = centred.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        z = centred / sd
        out.append((z.T @ z) / len(z))
    return np.stack(out)


def upper(matrices: np.ndarray, k: int = 1) -> np.ndarray:
    iu = np.triu_indices(matrices.shape[-1], k=k)
    return matrices[:, iu[0], iu[1]]


def static_features(timeseries: np.ndarray) -> np.ndarray:
    return upper(correlations(timeseries), k=0)


def dynamic_features(timeseries: np.ndarray) -> np.ndarray:
    """Mean and SD of sliding-window coupling.

    The SD is what static connectivity throws away: two participants can share a
    time-averaged coupling while differing in how much it fluctuates.
    """
    n_timepoints = timeseries.shape[1]
    starts = range(0, n_timepoints - WINDOW + 1, STRIDE)
    windows = np.stack([upper(correlations(timeseries[:, s:s + WINDOW, :]))
                        for s in starts])          # (n_windows, n_subj, n_edges)
    return np.hstack([windows.mean(axis=0), windows.std(axis=0)])


def strength_features(timeseries: np.ndarray, networks: np.ndarray | None) -> np.ndarray:
    """Positive and negative weighted node strength, averaged within network.

    Splitting by sign matters: a region that is strongly coupled to some areas
    and strongly anti-coupled to others has a near-zero signed strength but is
    not weakly connected.
    """
    matrices = correlations(timeseries)
    n = matrices.shape[-1]
    off = ~np.eye(n, dtype=bool)
    stacked = np.stack([m[off].reshape(n, n - 1) for m in matrices])
    positive = np.clip(stacked, 0, None).mean(axis=2)
    negative = np.clip(stacked, None, 0).mean(axis=2)

    if networks is None:
        return np.hstack([positive, negative])
    out = []
    for block in (positive, negative):
        out.append(np.stack([block[:, networks == net].mean(axis=1)
                             for net in np.unique(networks)], axis=1))
    return np.hstack(out)


def evaluate(x: np.ndarray, y: np.ndarray, splits, classifier: str,
             reduce: str | None = None, k: int = 20) -> float:
    """Mean accuracy, with every fitted step fitted inside the training fold."""
    steps: list = [('scale', StandardScaler())]
    if reduce == 'pca':
        steps.append(('reduce', PCA(n_components=min(k, 24), random_state=0)))
    elif reduce == 'select':
        steps.append(('reduce', SelectKBest(f_classif, k=min(k, x.shape[1]))))

    folds = []
    for train, test in splits:
        model = Pipeline([*steps, ('clf', _make_classifier(classifier, 0))])
        model.fit(x[train], y[train])
        folds.append(float((model.predict(x[test]) == y[test]).mean()))
    return float(np.mean(folds))


def families_for(atlas: str) -> dict[str, tuple[np.ndarray, str | None, int]]:
    """(features, in-fold reduction, k) per family, for one atlas."""
    dataset = load(atlas)
    region = dataset['timeseries'].transpose('subject', 'timepoint', 'region').values
    has_networks = 'network' in dataset.variables

    out: dict[str, tuple[np.ndarray, str | None, int]] = {}
    if has_networks:
        aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
        network = aggregated['timeseries'].transpose(
            'subject', 'timepoint', 'network').values
        out['static'] = (static_features(network), None, 0)
        out['dynamic'] = (dynamic_features(network), None, 0)
        out['strength'] = (
            strength_features(region, np.asarray(dataset['network'].values)), None, 0)
    else:
        out['static'] = (static_features(region), 'pca', 20)

    # Region scale, reduced inside the folds rather than dismissed. Capped: a
    # 1000-parcel atlas has ~500k edges, and the sliding-window version holds
    # (n_windows, n_subjects, n_edges) in memory at once, which is tens of GB
    # for no plausible benefit -- the 9-atlas run already showed accuracy
    # *falling* with resolution. Skips are reported rather than silent.
    n_regions = region.shape[2]
    if n_regions <= REGION_CAP:
        region_edges = upper(correlations(region))
        out['region+pca'] = (region_edges, 'pca', 20)
        out['region+select'] = (region_edges, 'select', 40)
    if has_networks and n_regions <= DYNAMIC_CAP:
        out['dynamic+pca'] = (dynamic_features(region), 'pca', 20)
    return out


def run_atlas(atlas: str, splits, seed: int) -> list[dict]:
    try:
        families = families_for(atlas)
    except Exception as error:
        return [{'atlas': atlas, 'family': 'FAILED', 'classifier': '-',
                 'accuracy': float('nan'), 'n_features': 0,
                 'error': f'{type(error).__name__}: {error}'}]

    dataset = load(atlas)
    y = labels_of(dataset)
    rows = []
    for family, (x, reduce, k) in families.items():
        for classifier in CLASSIFIERS:
            started = time.perf_counter()
            accuracy = evaluate(x, y, splits, classifier, reduce, k)
            interval = accuracy_interval(accuracy * len(y), len(y))
            rows.append({'atlas': atlas, 'family': family, 'classifier': classifier,
                         'accuracy': accuracy, 'n_features': int(x.shape[1]),
                         'reduce': reduce or '-',
                         'wilson_low': interval.low, 'wilson_high': interval.high,
                         'seconds': round(time.perf_counter() - started, 1),
                         'error': ''})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlases', nargs='*', default=None)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=25)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    atlases = args.atlases or sorted(
        p.name.removeprefix('timeseries_').removesuffix('.nc5')
        for p in CACHE.glob('timeseries_*.nc5'))

    splits = fixed_splits(32, args.n_splits, args.n_repeats, args.seed)
    print(f'{len(atlases)} atlases x feature families x {len(CLASSIFIERS)} classifiers, '
          f'{args.n_splits} x {args.n_repeats} CV\n')

    batches = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_atlas)(atlas, splits, args.seed) for atlas in atlases)
    table = pd.DataFrame([row for batch in batches for row in batch])
    table = table.sort_values('accuracy', ascending=False)
    table.to_csv(OUT / 'exploration.csv', index=False)

    failures = table[table['family'] == 'FAILED']
    for _, row in failures.iterrows():
        print(f'  FAILED {row["atlas"]}: {row["error"]}')
    table = table[table['family'] != 'FAILED']

    print('\ntop 15 of', len(table), 'configurations searched:')
    for _, r in table.head(15).iterrows():
        print(f'  {r["accuracy"]:.3f}  [{r["wilson_low"]:.2f}, {r["wilson_high"]:.2f}]  '
              f'{r["atlas"]:26s} {r["family"]:14s} {r["classifier"]:8s} '
              f'({r["n_features"]} feat)')

    print('\nbest per family:')
    for family, group in table.groupby('family'):
        best = group.loc[group['accuracy'].idxmax()]
        print(f'  {family:14s} {best["accuracy"]:.3f}  '
              f'{best["atlas"]} / {best["classifier"]}')

    print('\nbest per atlas:')
    for atlas, group in table.groupby('atlas'):
        best = group.loc[group['accuracy'].idxmax()]
        print(f'  {atlas:26s} {best["accuracy"]:.3f}  '
              f'{best["family"]} / {best["classifier"]}')

    (OUT / 'exploration.json').write_text(json.dumps({
        'n_configurations_searched': len(table),
        'cv': {'n_splits': args.n_splits, 'n_repeats': args.n_repeats},
        'best': table.iloc[0].to_dict() if len(table) else None,
        'warning': 'a maximum over a search; carries no inferential weight until '
                   'run through the family-wise null',
    }, indent=2, default=float))
    print(f'\n{len(table)} configurations searched. Written to {OUT}/.')
    print('NOTE: the maximum of a search is not a result. Anything promising here '
          'must go into run_permutation.py and clear the family-wise null.')


if __name__ == '__main__':
    main()
