"""Is the multiverse's attribution consensus predictable without the labels?

The companion null to `run_attribution_null.py`, and the cheaper half of the same
question. Haufe's forward pattern is A = Sigma w / (w' Sigma w). As the decoder
weight w degenerates towards noise -- which is the regime a chance-level
classifier is in -- A degenerates towards Sigma w, whose ranking is set by the
feature covariance. Sigma does not depend on the labels and is shared by every
specification that uses the same statistic, so a sceptic can propose that the
agreement reported in section 3.4.3 is a fact about the covariance of these 21
edges rather than a fact about action video game players.

That proposal is directly checkable and costs no model fitting at all. Rank the
edges by label-free properties of the feature matrix, and see whether those
rankings reproduce the observed multiverse consensus:

    variance              spread across participants
    mean |value|          typical magnitude
    top eigenvector       loading on the leading direction of Sigma
    random-decoder Haufe  mean |Sigma w| over random draws of w, i.e. exactly
                          the pattern a chance-level linear decoder produces

Each is compared to the consensus by the same rank-biased overlap used
throughout, against the same random-ranking floor. Labels are loaded only to
assert that they are never used.

Usage:
    uv run python scripts/run_attribution_geometry.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acnets.connectome import ExtraConnectivityMeasure
from acnets.multiverse.agreement import rank_biased_overlap
from acnets.multiverse.runner import edge_names, vectorize
from acnets.pipeline import TimeseriesAggregator

DATA = Path('data/Julia2018')
OUT = Path('outputs/agreement')

STATISTICS = ('correlation', 'partial correlation', 'covariance', 'precision',
              'tangent', 'chatterjee')


def label_free_rankings(x: np.ndarray,
                        names: list[str],
                        n_draws: int,
                        seed: int) -> dict[str, pd.Series]:
    """Four orderings of the same features, none of which has seen a label."""
    cov = np.cov(x - x.mean(axis=0), rowvar=False)
    rng = np.random.default_rng(seed)
    random_haufe = np.zeros(len(names))
    for _ in range(n_draws):
        random_haufe += np.abs(cov @ rng.standard_normal(len(names)))

    return {
        'variance': pd.Series(x.var(axis=0, ddof=1), index=names),
        'mean |value|': pd.Series(np.abs(x).mean(axis=0), index=names),
        'top eigenvector': pd.Series(np.abs(np.linalg.eigh(cov)[1][:, -1]),
                                     index=names),
        'random-decoder Haufe': pd.Series(random_haufe / n_draws, index=names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-draws', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dataset = xr.open_dataset(DATA / 'timeseries_dosenbach2010.nc5',
                              engine='h5netcdf').load()
    np.random.seed(args.seed)
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    timeseries = aggregated['timeseries'].transpose(
        'subject', 'timepoint', 'network').values
    nodes = [str(n) for n in aggregated['network'].values]
    names = edge_names(nodes, asymmetric=False)

    consensus = list(
        pd.read_csv(OUT / 'selection_frequency_network.csv', index_col=0)
        .iloc[:, 0].sort_values(ascending=False).index)

    print(f'{timeseries.shape[0]} subjects, {len(nodes)} networks, '
          f'{len(names)} edges; consensus over {len(consensus)} ranked features')

    rows = []
    for statistic in STATISTICS:
        measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
        measure.fit(timeseries)
        x = vectorize(measure.transform(timeseries), False)

        for criterion, series in label_free_rankings(
                x, names, args.n_draws, args.seed).items():
            order = list(series.sort_values(ascending=False).index)
            rows.append({
                'statistic': statistic,
                'criterion': criterion,
                'rank_CON_SMN': order.index('CON ↔ SMN') + 1,
                'rank_FPN_SMN': order.index('FPN ↔ SMN') + 1,
                'rbo_vs_consensus': rank_biased_overlap(order, consensus, p=0.9),
                'top3': ', '.join(order[:3]),
            })

    table = pd.DataFrame(rows)

    rng = np.random.default_rng(args.seed)
    floor = np.array([rank_biased_overlap(list(rng.permutation(names)),
                                          consensus, p=0.9)
                      for _ in range(2000)])

    pd.set_option('display.width', 200, 'display.max_colwidth', 60)
    print()
    print(table.round(3).to_string(index=False))
    print(f'\nrandom-ranking floor vs consensus: {floor.mean():.3f} '
          f'[{np.percentile(floor, 2.5):.3f}, {np.percentile(floor, 97.5):.3f}]')
    print(f'label-free rankings span            '
          f'{table.rbo_vs_consensus.min():.3f} - {table.rbo_vs_consensus.max():.3f}')
    print(f'CON-SMN best label-free rank        '
          f'{table.rank_CON_SMN.min()} of {len(names)}')

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / 'attribution_geometry.csv', index=False)
    print(f'\nwrote {OUT / "attribution_geometry.csv"}')


if __name__ == '__main__':
    main()
