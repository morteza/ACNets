"""Pretraining that targets reliability, the constraint that actually binds.

LEMON pretraining has been tried three ways in this project and helped none of
them: as a tangent reference and scaler (1.6% of accuracy variance), as a
per-timepoint autoencoder (the pre-registered primary, 0.477), and as a
diffusion normative model (at chance). All three used external data to improve
*estimation* -- a better covariance reference, a better representation -- and
estimation is not what limits this design. Section 11 shows the binding ceiling
is measurement reliability: at ICC = 0.29, no latent effect short of d = 3.1
can express itself as 80% accuracy.

So this asks a different question of the external data: **which edges are
measured reliably, and can that be used?**

Two things follow, and the first matters more than the second.

**1. Measure the reliability rather than citing it.** The ceiling analysis
borrowed ICC = 0.29 from Noble, Scheinost and Constable (2019). That is a
meta-analytic average over other datasets, other scanners and other run lengths.
It can be estimated directly here by splitting each participant's run in half,
computing connectivity on each half, and correlating the halves across
participants -- with a Spearman-Brown correction back up to full run length.
Doing this on both Julia2018 and LEMON says whether the ceiling analysis rests
on the right number.

**2. Weight features by how reliably they are measured.** Reliability estimated
on LEMON is external to the target sample and label-free, so it can scale the
target's features without leakage. Down-weighting unreliable edges is
principled: an edge measured with ICC 0.05 contributes almost pure noise to a
discriminant, and a classifier that treats it like a reliable edge is wasting
its limited degrees of freedom.

A caveat that decides the design: *uniform* shrinkage toward a population mean
is an affine transform and leaves every linear classifier's accuracy exactly
unchanged. Only *per-edge* weighting does anything. That is why this is a
diagonal rescaling by reliability rather than shrinkage toward a prior, which is
what the earlier LEMON work already did.

    uv run python scripts/run_reliability_transfer.py --n-permutations 500
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
import xarray as xr  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.covariance import OAS  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure  # noqa: E402
from acnets.multiverse import accuracy_interval, max_stat_permutation  # noqa: E402
from acnets.multiverse.runner import edge_names, vectorize  # noqa: E402
from acnets.pipeline import TimeseriesAggregator  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402

OUT = Path('outputs/reliability')
LEMON = Path('data/LEMON/timeseries_dosenbach2010.nc')
STATISTICS = ('correlation', 'partial correlation', 'precision', 'tangent')


def connectivity(timeseries: np.ndarray, statistic: str) -> np.ndarray:
    measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
    measure.fit(timeseries)
    return vectorize(measure.transform(timeseries), False)


def split_half_reliability(timeseries: np.ndarray, statistic: str
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge split-half reliability, and its Spearman-Brown correction.

    Each participant's run is cut in half, connectivity computed on each half,
    and the two halves correlated across participants. That correlation is the
    reliability of a *half-length* run; Spearman-Brown, r_full = 2r / (1 + r),
    projects it back to the full run.
    """
    middle = timeseries.shape[1] // 2
    first = connectivity(timeseries[:, :middle, :], statistic)
    second = connectivity(timeseries[:, middle:, :], statistic)

    half = np.zeros(first.shape[1])
    for j in range(first.shape[1]):
        if np.ptp(first[:, j]) < 1e-12 or np.ptp(second[:, j]) < 1e-12:
            continue                      # constant edge, e.g. a fixed diagonal
        half[j] = stats.pearsonr(first[:, j], second[:, j]).statistic
    half = np.clip(half, 0.0, 0.999)
    full = 2 * half / (1 + half)
    return half, full


def lemon_timeseries(window: int) -> np.ndarray:
    """LEMON at network scale, truncated to the target's run length."""
    dataset = xr.open_dataset(LEMON, engine='h5netcdf').load()
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    values = aggregated['timeseries'].transpose(
        'subject', 'timepoint', 'network').values
    return values[:, :window, :]


def loocv(x: np.ndarray, y: np.ndarray,
          weights: np.ndarray | None = None) -> float:
    """The best estimator found in section 12: shrinkage LDA under LOOCV.

    `weights` are applied *after* standardisation, which is the only place they
    can survive. Applying them to the raw features instead is a no-op: the
    scaler divides each column by its own standard deviation and so exactly
    undoes any diagonal rescaling. The first version of this script did that and
    produced weighted results identical to the unweighted ones to three decimal
    places, which is what gave the mistake away.
    """
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        x_train = scaler.transform(x[train])
        x_test = scaler.transform(x[test])
        if weights is not None:
            x_train = x_train * weights
            x_test = x_test * weights
        model = LinearDiscriminantAnalysis(solver='lsqr',
                                           covariance_estimator=OAS())
        model.fit(x_train, y[train])
        predictions[test] = model.predict(x_test)
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
    target_ts = timeseries['network']
    names = edge_names(nodes['network'], False)
    window = target_ts.shape[1]

    external_ts = lemon_timeseries(window)
    print(f'target {target_ts.shape}, LEMON truncated to {external_ts.shape}\n')

    print('=' * 70)
    print('MEASURED RELIABILITY (split-half, Spearman-Brown corrected)')
    print('=' * 70)
    print(f'{"statistic":22s} {"Julia2018":>18s} {"LEMON":>18s}')
    reliabilities, payload_rel = {}, {}
    for statistic in STATISTICS:
        _, target_full = split_half_reliability(target_ts, statistic)
        _, external_full = split_half_reliability(external_ts, statistic)
        reliabilities[statistic] = external_full
        varying = [j for j in range(len(names)) if external_full[j] > 0]
        payload_rel[statistic] = {
            'julia2018_median': float(np.median(target_full[varying])),
            'lemon_median': float(np.median(external_full[varying])),
            'lemon_per_edge': {names[j]: float(external_full[j]) for j in varying},
        }
        print(f'{statistic:22s} {np.median(target_full[varying]):18.3f} '
              f'{np.median(external_full[varying]):18.3f}')

    literature = 0.29
    measured = np.median([v['lemon_median'] for v in payload_rel.values()])
    print(f'\n  literature value used in section 11: {literature:.2f}')
    print(f'  measured here (median over statistics, LEMON): {measured:.2f}')
    print(f'  the ceiling analysis is '
          f'{"conservative" if measured > literature else "optimistic"} by this'
          f' measure')

    print()
    print('=' * 70)
    print('DOES RELIABILITY WEIGHTING HELP?')
    print('=' * 70)
    observed = {}
    for statistic in STATISTICS:
        x = connectivity(target_ts, statistic)
        weights = reliabilities[statistic]
        observed[f'{statistic}/plain'] = loocv(x, y)
        # weight by reliability: an edge measured at ICC 0.05 is nearly noise
        observed[f'{statistic}/reliability-weighted'] = loocv(x, y, weights)
        # and the harsher variant, keeping only the reliably measured edges
        keep = weights >= np.median(weights[weights > 0])
        if keep.sum() >= 2:
            observed[f'{statistic}/reliable-edges-only'] = loocv(x[:, keep], y)

    table = pd.DataFrame([{'cell': k, 'accuracy': v}
                          for k, v in observed.items()]).sort_values(
        'accuracy', ascending=False)
    for _, row in table.iterrows():
        interval = accuracy_interval(row['accuracy'] * len(y), len(y))
        print(f'  {row["cell"]:46s} {row["accuracy"]:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

    started = time.perf_counter()
    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}:')

    def one(i: int) -> list[float]:
        rng = np.random.default_rng(args.seed + 1 + i)
        permuted = rng.permutation(y)
        scores = {}
        for statistic in STATISTICS:
            x = connectivity(target_ts, statistic)
            weights = reliabilities[statistic]
            scores[f'{statistic}/plain'] = loocv(x, permuted)
            scores[f'{statistic}/reliability-weighted'] = loocv(
                x, permuted, weights)
            keep = weights >= np.median(weights[weights > 0])
            if keep.sum() >= 2:
                scores[f'{statistic}/reliable-edges-only'] = loocv(x[:, keep],
                                                                   permuted)
        return [scores[k] for k in observed]

    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(one)(i) for i in range(args.n_permutations)))
    print(f'  {time.perf_counter() - started:.0f}s')

    result = max_stat_permutation(observed, null)
    print(f'\n  observed max        {result.observed_max:.3f}')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  null max, 95th pct  {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')
    clear = {k: v for k, v in observed.items() if v >= result.critical_value}
    print(f'  {len(clear)} of {len(observed)} cells clear it')

    (OUT / 'reliability.json').write_text(json.dumps({
        'reliability': payload_rel,
        'literature_icc': literature,
        'measured_icc_median': float(measured),
        'observed': observed,
        **result.summary(),
        'cells_clearing': clear,
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
