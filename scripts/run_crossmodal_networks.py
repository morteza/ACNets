"""Discover networks in one modality and dataset, test them in the other.

Only the **identity of the networks** crosses the gap -- no features, no model,
no weights. Direction A ranks Schaefer parcels by attribution on gong2015's
*structural* classifier, keeps the networks those parcels belong to, and asks
whether restricting Julia2018's *functional* connectivity to them helps.
Direction B does the reverse.

This is the strongest cross-dataset test the available data support. gong2015
has no resting-state fMRI, so a functional result cannot be replicated in it
directly; but a claim about *which networks matter* is modality-independent by
construction, and can therefore be discovered in one modality and tested in the
other.

**Baseline.** Restricting to a subset of networks reduces dimensionality, which
helps by itself at these sample sizes. So the comparison is against **random
network subsets of the same size**, enumerated exhaustively -- there are only
C(7,3) = 35 three-network subsets, so the null is exact rather than sampled.
The question is whether *these* networks beat *any* networks.

**A label bug this script has to route around.** nilearn returns 201 Schaefer
names for 200 parcels with 'Background' first, and both the cached functional
timeseries and the earlier atlas tables inherited the resulting off-by-one -- the
region coordinate at position 0 is called 'Background' but is really parcel 1.
Network membership is therefore taken from the corrected atlas CSV by *position*
(index i is label i+1), never from the stored names.

    uv run python scripts/run_crossmodal_networks.py --n-networks 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from sklearn.covariance import OAS  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.explain.haufe import haufe_transform  # noqa: E402
from acnets.multiverse import accuracy_interval  # noqa: E402
from run_reliability_transfer import connectivity  # noqa: E402

OUT = Path('outputs/crossmodal_networks')
ATLAS_DIR = Path('data/atlases/prepared')
FUNCTIONAL = Path('data/Julia2018/derivatives/resting_timeseries/'
                  'timeseries_dosenbach2010.nc5')
MAPPING = ATLAS_DIR / 'schaefer200_to_dosenbach.csv'

# the structural parcels are Schaefer (volumetric, reliable across datasets --
# section 19) but are *labelled* with Dosenbach networks by nearest-ROI
# centroid, so both modalities speak the same network vocabulary. Full names in
# the Dosenbach table, abbreviations in the cached timeseries.
ABBREVIATION = {'default': 'DMN', 'sensorimotor': 'SMN',
                'cingulo-opercular': 'CON', 'fronto-parietal': 'FPN',
                'occipital': 'VIS', 'cerebellum': 'CER'}
STRUCTURAL = Path('data/gong2015/derivatives/gmv_gong2015_schaefer200.csv')


def parcel_networks() -> list[str]:
    """Dosenbach network per Schaefer parcel, by label order."""
    mapping = pd.read_csv(MAPPING).sort_values('label')
    return [ABBREVIATION.get(str(n), 'NA') for n in mapping['dosenbach_network']]


def structural_source():
    table = pd.read_csv(STRUCTURAL)
    columns = [c for c in table.columns
               if c not in ('subject', 'total_gm_mm3', 'brain_mask_cm3')]
    x = table[columns].to_numpy(float)
    y = np.array([0 if str(s).startswith('expert') else 1
                  for s in table['subject']])
    return x, y, [int(c) for c in columns]


def functional_timeseries():
    """Julia2018 at Dosenbach scale, with its own native network assignment."""
    dataset = xr.open_dataset(FUNCTIONAL)
    values = dataset['timeseries'].transpose(
        'subject', 'timepoint', 'region').values
    y = np.array([0 if 'AVGP' in str(s) else 1
                  for s in dataset['subject'].values])
    region_networks = [str(n) for n in np.asarray(dataset['network'].values)]
    return values, y, region_networks


def loocv(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or x.shape[1] < 2:
        return float('nan')
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        model = LinearDiscriminantAnalysis(solver='lsqr',
                                           covariance_estimator=OAS())
        model.fit(scaler.transform(x[train]), y[train])
        predictions[test] = model.predict(scaler.transform(x[test]))
    return float((predictions == y).mean())


def network_connectivity(timeseries, networks, keep) -> np.ndarray:
    """Aggregate parcels to the kept networks, then network-scale connectivity."""
    order = sorted(keep)
    aggregated = np.stack(
        [timeseries[:, :, [i for i, n in enumerate(networks) if n == net]].mean(axis=2)
         for net in order], axis=2)
    return connectivity(aggregated, 'precision')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-networks', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=40,
                        help='parcels kept from the attribution ranking before '
                             'counting which networks they belong to')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    networks = parcel_networks()
    unique = sorted({n for n in networks if n != 'nan'})
    print(f'{len(networks)} parcels over {len(unique)} networks: '
          f'{", ".join(unique)}\n')

    struct_x, struct_y, struct_labels = structural_source()
    func_ts, func_y, func_networks = functional_timeseries()
    print(f'gong2015 structural  n={len(struct_y)}, {struct_x.shape[1]} parcels')
    print(f'Julia2018 functional n={len(func_y)}, {func_ts.shape[2]} parcels\n')

    payload = {}

    # ---- Direction A: discover on structural, test on functional -----------
    scaler = StandardScaler().fit(struct_x)
    scaled = scaler.transform(struct_x)
    fitted = LinearDiscriminantAnalysis(
        solver='lsqr', covariance_estimator=OAS()).fit(scaled, struct_y)
    ranking = np.abs(haufe_transform(scaled, fitted.coef_.ravel()))
    top = np.argsort(ranking)[::-1][:args.top_k]
    counts = Counter(networks[struct_labels[i] - 1] for i in top)
    discovered = [n for n, _ in counts.most_common(args.n_networks)]
    print(f'A. gong2015 STRUCTURAL attribution -> top {args.top_k} parcels')
    print('   network composition: '
          + ', '.join(f'{k} {v}' for k, v in counts.most_common(5)))
    print(f'   discovered networks: {"+".join(sorted(discovered))}')

    shared_networks = sorted(set(unique) & set(func_networks))
    discovered = [n for n in discovered if n in shared_networks]
    observed = loocv(network_connectivity(func_ts, func_networks, discovered),
                     func_y)
    subsets = list(combinations(shared_networks, args.n_networks))
    everything = np.array([loocv(
        network_connectivity(func_ts, func_networks, list(s)), func_y)
        for s in subsets])
    rank = int((everything > observed).sum()) + 1
    p = float((1 + (everything >= observed).sum()) / (len(everything) + 1))
    interval = accuracy_interval(observed * len(func_y), len(func_y))
    print(f'   tested on Julia2018 FUNCTIONAL: {observed:.3f} '
          f'[{interval.low:.2f}, {interval.high:.2f}]')
    print(f'   against all {len(subsets)} network triples: rank {rank}, '
          f'median {np.median(everything):.3f}, best {everything.max():.3f}')
    print(f'   p = {p:.4f}'
          f'{"   TRANSFERS" if p < 0.05 else "   does not transfer"}\n')
    payload['structural->functional'] = {
        'discovered_networks': sorted(discovered),
        'network_counts': dict(counts.most_common()),
        'accuracy': observed, 'rank_of_subsets': rank,
        'n_subsets': len(subsets), 'p': p,
        'subset_median': float(np.median(everything)),
        'subset_best': float(everything.max())}

    # ---- Direction B: discover on functional, test on structural -----------
    full = network_connectivity(func_ts, func_networks, shared_networks)
    scaler = StandardScaler().fit(full)
    scaled = scaler.transform(full)
    fitted = LinearDiscriminantAnalysis(
        solver='lsqr', covariance_estimator=OAS()).fit(scaled, func_y)
    edge_weight = np.abs(haufe_transform(scaled, fitted.coef_.ravel()))
    # map edge attribution back onto networks: a network's score is the total
    # attribution of every edge it participates in
    order = sorted(shared_networks)
    iu = np.triu_indices(len(order))
    score = Counter()
    for weight, a, b in zip(edge_weight, iu[0], iu[1], strict=True):
        score[order[a]] += weight
        score[order[b]] += weight
    discovered_b = [n for n, _ in score.most_common(args.n_networks)]
    print('B. Julia2018 FUNCTIONAL attribution -> network scores')
    print('   ' + ', '.join(f'{k} {v:.2f}' for k, v in score.most_common(5)))
    print(f'   discovered networks: {"+".join(sorted(discovered_b))}')

    def structural_subset(keep):
        cols = [i for i, label in enumerate(struct_labels)
                if networks[label - 1] in keep]
        return struct_x[:, cols]

    observed_b = loocv(structural_subset(discovered_b), struct_y)
    everything_b = np.array([loocv(structural_subset(list(s)), struct_y)
                             for s in subsets])
    rank_b = int((everything_b > observed_b).sum()) + 1
    p_b = float((1 + (everything_b >= observed_b).sum()) / (len(everything_b) + 1))
    interval_b = accuracy_interval(observed_b * len(struct_y), len(struct_y))
    print(f'   tested on gong2015 STRUCTURAL: {observed_b:.3f} '
          f'[{interval_b.low:.2f}, {interval_b.high:.2f}]')
    print(f'   against all {len(subsets)} network triples: rank {rank_b}, '
          f'median {np.median(everything_b):.3f}, best {everything_b.max():.3f}')
    print(f'   p = {p_b:.4f}'
          f'{"   TRANSFERS" if p_b < 0.05 else "   does not transfer"}')
    payload['functional->structural'] = {
        'discovered_networks': sorted(discovered_b),
        'network_scores': {k: float(v) for k, v in score.most_common()},
        'accuracy': observed_b, 'rank_of_subsets': rank_b,
        'n_subsets': len(subsets), 'p': p_b,
        'subset_median': float(np.median(everything_b)),
        'subset_best': float(everything_b.max())}

    (OUT / 'crossmodal_networks.json').write_text(json.dumps({
        'n_networks': args.n_networks, 'top_k_parcels': args.top_k,
        'directions': payload,
        'note': 'only network identity crosses datasets; features, models and '
                'weights are re-estimated in the target modality',
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
