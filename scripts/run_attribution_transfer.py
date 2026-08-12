"""Discover regions by explainable AI on one dataset, test them on the other.

Both directions, one modality (structural grey-matter volume), so the comparison
is like-for-like:

    gong2015 -> Julia2018    rank regions by attribution on gong2015's
                             expert/non-expert classifier, restrict Julia2018 to
                             the top k, and see whether that helps
    Julia2018 -> gong2015    the same in reverse

**This is a direct test of the project's central claim.** Section 3.4.3 argues
attribution is more stable than accuracy -- that a model can lean on the same
features reliably while never predicting well enough to beat a corrected null.
If that is true in a way that matters, an attribution ranking derived from a
*chance-level* classifier on one dataset should still carry information about
which regions matter in another. If it is false, the ranking is noise and
transfers nothing. Structural decoding is at chance in both datasets (sections
20-21), so this is the cleanest available test of exactly that proposition.

**The baseline is what makes it a test.** Restricting to k regions out of 200
usually helps by itself, because 200 features against 32-47 participants is
hopeless. So the comparison is not against all-parcels but against **random sets
of k regions**, drawn many times. The question is whether *these* k regions beat
*any* k regions, which is the only version that says something about
localisation rather than dimensionality.

Attribution is the Haufe forward transform of a linear discriminant -- the same
channel the paper uses for its linear specifications, and the one that is a
statement about the data rather than about the classifier's weights.

    uv run python scripts/run_attribution_transfer.py --top-k 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.covariance import OAS  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.explain.haufe import haufe_transform  # noqa: E402
from acnets.multiverse import accuracy_interval  # noqa: E402

OUT = Path('outputs/attribution_transfer')
ATLAS_DIR = Path('data/atlases/prepared')


def load(atlas: str):
    frames = {}
    for name, path, rule in (
            ('gong2015',
             f'data/gong2015/derivatives/gmv_gong2015_{atlas}.csv',
             lambda s: 0 if s.startswith('expert') else 1),
            ('julia2018',
             f'data/Julia2018/derivatives/gmv_julia2018_{atlas}.csv',
             lambda s: 0 if 'AVGP' in s else 1)):
        table = pd.read_csv(path)
        columns = [c for c in table.columns
                   if c not in ('subject', 'total_gm_mm3', 'brain_mask_cm3')]
        frames[name] = (table[columns].to_numpy(float),
                        np.array([rule(str(s)) for s in table['subject']]),
                        columns)
    a, b = frames['gong2015'], frames['julia2018']
    shared = [c for c in a[2] if c in b[2]]
    index_a = [a[2].index(c) for c in shared]
    index_b = [b[2].index(c) for c in shared]
    return ({'gong2015': (a[0][:, index_a], a[1]),
             'julia2018': (b[0][:, index_b], b[1])}, shared)


def attribution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Haufe forward pattern of a shrinkage discriminant, fitted on all rows.

    Fitting on the whole *source* dataset is correct here: the source is the
    discovery sample and is never predicted. What must not leak is the target,
    and the target is untouched by this function.
    """
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    model = LinearDiscriminantAnalysis(solver='lsqr',
                                       covariance_estimator=OAS()).fit(scaled, y)
    return np.abs(haufe_transform(scaled, model.coef_.ravel()))


def loocv(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[1] < 2:
        return float('nan')
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        model = LinearDiscriminantAnalysis(solver='lsqr',
                                           covariance_estimator=OAS())
        model.fit(scaler.transform(x[train]), y[train])
        predictions[test] = model.predict(scaler.transform(x[test]))
    return float((predictions == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='schaefer200')
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--n-random', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data, shared = load(args.atlas)
    labels = pd.read_csv(ATLAS_DIR / f'{args.atlas}_labels.csv')
    network_of = {str(row.label): row.network for row in labels.itertuples()}
    print(f'{len(shared)} shared regions ({args.atlas}); '
          f'top-{args.top_k} carried across\n')

    payload = {}
    for source, target in (('gong2015', 'julia2018'),
                           ('julia2018', 'gong2015')):
        sx, sy = data[source]
        tx, ty = data[target]

        ranking = attribution(sx, sy)
        chosen = np.argsort(ranking)[::-1][:args.top_k]
        networks = Counter(network_of.get(shared[i], 'NA') for i in chosen)

        discovered = loocv(tx[:, chosen], ty)
        everything = loocv(tx, ty)

        rng = np.random.default_rng(args.seed)
        random_sets = np.array(Parallel(n_jobs=args.n_jobs)(
            delayed(loocv)(tx[:, rng.choice(len(shared), args.top_k,
                                            replace=False)], ty)
            for _ in range(args.n_random)))
        random_sets = random_sets[np.isfinite(random_sets)]
        p = float((1 + (random_sets >= discovered).sum())
                  / (len(random_sets) + 1))

        interval = accuracy_interval(discovered * len(ty), len(ty))
        print(f'{source} -> {target}   (discover on {source}, n={len(sy)}; '
              f'test on {target}, n={len(ty)})')
        print(f'  top-{args.top_k} networks: '
              + ', '.join(f'{k} {v}' for k, v in networks.most_common(4)))
        print(f'  discovered regions   {discovered:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]')
        print(f'  all {len(shared)} regions     {everything:.3f}')
        print(f'  random {args.top_k}-region sets  median '
              f'{np.median(random_sets):.3f}, 95th pct '
              f'{np.percentile(random_sets, 95):.3f}')
        print(f'  p vs random sets     {p:.4f}'
              f'{"   TRANSFERS" if p < 0.05 else "   does not transfer"}\n')

        payload[f'{source}->{target}'] = {
            'discovered': discovered, 'all_regions': everything,
            'random_median': float(np.median(random_sets)),
            'random_p95': float(np.percentile(random_sets, 95)),
            'p_vs_random': p,
            'top_networks': dict(networks.most_common()),
            'n_source': len(sy), 'n_target': len(ty)}

    (OUT / f'attribution_transfer_{args.atlas}.json').write_text(json.dumps({
        'atlas': args.atlas, 'top_k': args.top_k,
        'modality': 'grey-matter volume',
        'directions': payload,
        'interpretation': 'the baseline is random region sets of the same size, '
                          'not all-regions, because restricting dimensionality '
                          'helps by itself at this sample size',
    }, indent=2, default=float))
    print(f'written to {OUT}/')


if __name__ == '__main__':
    main()
