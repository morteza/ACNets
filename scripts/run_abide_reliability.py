"""Estimate per-edge reliability from a large open corpus, and use it.

Section 13 showed that transferring *reliability* from external data works where
transferring representations did not (+0.031, the project's best honest result),
and that the limit on it is how precisely reliability can be estimated. LEMON has
220 participants. This uses ABIDE, which has roughly five times as many and --
crucially -- ships timeseries already extracted with the **Dosenbach 160 atlas**,
the same parcellation this project uses.

**Why not HCP.** HCP is the obvious suggestion and it is the wrong corpus for
this specific job, for two reasons. Access needs a ConnectomeDB account and an
accepted data use agreement, which are the user's to obtain. More importantly,
reliability is a property of an *acquisition*, not of a brain: HCP is TR = 0.72 s,
1,200 timepoints, multiband. Its per-edge reliability is far higher and
structurally different from a TR = 3 s, 124-timepoint, 4 mm single-band scan, so
transferring it would import weights calibrated to a regime the target is not in.
ABIDE's sites run TR 1.5-3 s at 120-300 timepoints, which brackets the target.
A bigger corpus only helps if it measures the same quantity.

Only ABIDE's controls are used. A sample containing two deliberately contrasting
groups has extra between-participant variance, which inflates split-half
reliability -- the artefact already seen in Julia2018's own estimate (section
13.1). Controls are the unselected population the estimator assumes.

    uv run python scripts/run_abide_reliability.py --n-subjects 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from nilearn import datasets  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse import accuracy_interval, max_stat_permutation  # noqa: E402
from acnets.multiverse.runner import edge_names  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402
from run_reliability_transfer import (  # noqa: E402
    connectivity,
    lemon_timeseries,
    loocv,
    split_half_reliability,
)

warnings.filterwarnings('ignore')

OUT = Path('outputs/reliability')
STATISTICS = ('correlation', 'partial correlation', 'precision', 'tangent')


def abide_network_timeseries(n_subjects: int, window: int,
                             networks: np.ndarray) -> np.ndarray:
    """ABIDE controls, aggregated to networks, truncated to the target length.

    The ROI ordering is verified rather than assumed: ABIDE's columns are
    numbered `#1..#161` with no labels, so the mapping onto this project's
    region-to-network assignment rests on both using the canonical Dosenbach
    ordering. The check is that within-network coupling must exceed
    between-network coupling, which is true of any real resting data and false
    under a scrambled ordering.
    """
    bundle = datasets.fetch_abide_pcp(
        derivatives=['rois_dosenbach160'], n_subjects=n_subjects,
        pipeline='cpac', quality_checked=True, verbose=0, DX_GROUP=2)
    runs = [np.asarray(t)[:, :160] for t in bundle['rois_dosenbach160']]
    usable = [r for r in runs if r.shape[0] >= window and np.isfinite(r).all()]
    print(f'ABIDE controls: {len(runs)} fetched, {len(usable)} with '
          f'>= {window} timepoints')
    if len(usable) < 50:
        raise SystemExit('too few usable ABIDE runs')

    stacked = np.stack([r[:window] for r in usable])
    _verify_ordering(stacked, networks)

    order = sorted(np.unique(networks))
    return np.stack([stacked[:, :, networks == net].mean(axis=2)
                     for net in order], axis=2)


def _verify_ordering(region_timeseries: np.ndarray, networks: np.ndarray) -> None:
    sample = region_timeseries[:20]
    matrices = []
    for run in sample:
        z = (run - run.mean(0)) / np.where(run.std(0) > 0, run.std(0), 1)
        matrices.append(z.T @ z / len(z))
    mean = np.mean(matrices, axis=0)

    unique = np.unique(networks)
    within = np.mean([
        mean[np.ix_(networks == n, networks == n)][
            ~np.eye((networks == n).sum(), dtype=bool)].mean() for n in unique])
    between = np.mean([
        mean[np.ix_(networks == a, networks == b)].mean()
        for i, a in enumerate(unique) for b in unique[i + 1:]])
    print(f'  ordering check: within-network r = {within:+.3f}, '
          f'between = {between:+.3f}')
    if within <= between + 0.02:
        raise SystemExit(
            'ABIDE ROI ordering does not match this project\'s Dosenbach '
            'assignment: within-network coupling is not above between-network. '
            'Reliability weights derived from it would be meaningless.')


def precision_of_estimate(timeseries: np.ndarray, statistic: str,
                          sizes: list[int], n_draws: int = 20,
                          seed: int = 0) -> dict[int, float]:
    """How much does the reliability estimate move if you resample the corpus?

    This is the question a bigger corpus is supposed to answer. Reported as the
    median across edges of the standard deviation of the estimate over
    resamples: smaller is a more trustworthy weight vector.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for size in sizes:
        # `size == len(timeseries)` is degenerate: sampling n of n without
        # replacement returns the same set every draw, so the reported spread is
        # exactly zero by construction rather than because the estimate is
        # precise. Excluded rather than printed as a misleading 0.0000.
        if size >= len(timeseries):
            continue
        estimates = []
        for _ in range(n_draws):
            index = rng.choice(len(timeseries), size, replace=False)
            _, full = split_half_reliability(timeseries[index], statistic)
            estimates.append(full)
        out[size] = float(np.median(np.std(np.array(estimates), axis=0)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-subjects', type=int, default=400)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, args.seed)
    target = timeseries['network']
    names = edge_names(nodes['network'], False)
    window = target.shape[1]
    networks = np.asarray(dataset['network'].values)

    external = {
        'LEMON (220)': lemon_timeseries(window),
        'ABIDE controls': abide_network_timeseries(args.n_subjects, window,
                                                   networks),
    }
    print()

    print('=' * 72)
    print('RELIABILITY, TWO CORPORA')
    print('=' * 72)
    print(f'{"statistic":22s} ' + ' '.join(f'{k:>16s}' for k in external)
          + f' {"agreement r":>13s}')
    weights, payload = {}, {}
    for statistic in STATISTICS:
        estimates = {}
        for label, series in external.items():
            _, full = split_half_reliability(series, statistic)
            estimates[label] = full
        varying = [j for j in range(len(names))
                   if all(e[j] > 0 for e in estimates.values())]
        agreement = stats.pearsonr(
            *[np.array([e[j] for j in varying]) for e in estimates.values()]
        ).statistic if len(varying) > 2 else float('nan')
        weights[statistic] = estimates['ABIDE controls']
        payload[statistic] = {
            label: float(np.median(e[varying])) for label, e in estimates.items()}
        payload[statistic]['corpus_agreement_r'] = float(agreement)
        print(f'{statistic:22s} '
              + ' '.join(f'{np.median(e[varying]):16.3f}' for e in estimates.values())
              + f' {agreement:13.3f}')

    print('\n' + '=' * 72)
    print('DOES A BIGGER CORPUS GIVE A MORE PRECISE WEIGHT VECTOR?')
    print('=' * 72)
    abide = external['ABIDE controls']
    sizes = [50, 100, 200, 400, len(abide)]
    precision = precision_of_estimate(abide, 'precision', sorted(set(sizes)))
    print('  corpus n   sd of the per-edge reliability estimate')
    for size, sd in precision.items():
        print(f'  {size:8d}   {sd:.4f}')
    payload['estimate_precision'] = {str(k): v for k, v in precision.items()}

    print('\n' + '=' * 72)
    print('CLASSIFICATION WITH ABIDE-DERIVED WEIGHTS')
    print('=' * 72)
    observed = {}
    for statistic in STATISTICS:
        x = connectivity(target, statistic)
        observed[f'{statistic}/plain'] = loocv(x, y)
        observed[f'{statistic}/abide-weighted'] = loocv(x, y, weights[statistic])
    table = pd.DataFrame([{'cell': k, 'accuracy': v}
                          for k, v in observed.items()]).sort_values(
        'accuracy', ascending=False)
    for _, row in table.iterrows():
        interval = accuracy_interval(row['accuracy'] * len(y), len(y))
        print(f'  {row["cell"]:42s} {row["accuracy"]:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

    started = time.perf_counter()
    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}:')

    def one(i: int) -> list[float]:
        rng = np.random.default_rng(args.seed + 1 + i)
        permuted = rng.permutation(y)
        scores = {}
        for statistic in STATISTICS:
            x = connectivity(target, statistic)
            scores[f'{statistic}/plain'] = loocv(x, permuted)
            scores[f'{statistic}/abide-weighted'] = loocv(x, permuted,
                                                          weights[statistic])
        return [scores[k] for k in observed]

    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(one)(i) for i in range(args.n_permutations)))
    print(f'  {time.perf_counter() - started:.0f}s')

    result = max_stat_permutation(observed, null)
    print(f'\n  observed max        {result.observed_max:.3f}')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  null max, 95th pct  {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')

    payload_out = {'reliability': payload, 'observed': observed,
                   'n_abide': len(abide), **result.summary()}
    (OUT / 'abide.json').write_text(json.dumps(payload_out, indent=2, default=float))
    print(f'\nwritten to {OUT}/abide.json')


if __name__ == '__main__':
    main()
