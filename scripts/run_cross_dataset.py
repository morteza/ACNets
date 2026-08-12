"""External validation: train on one dataset, predict the other.

This is the analysis the whole structural detour was for. A model fitted on
gong2015 and evaluated on Julia2018 (or the reverse) makes a single prediction
on a genuinely independent sample -- different participants, different scanner,
different site. It needs **no multiverse correction**, because nothing about the
held-out dataset was consulted while fitting, and it is therefore the one route
that converts a Tier 3 claim into a Tier 1 one without pre-registration
(`docs/paper/publication_strategy.md`).

Volumetric atlases only. Under the sphere-based Dosenbach parcellation the two
datasets' regional profiles correlate at **-0.41**; under AAL, **+0.87**
(section 19). Running this on a sphere atlas would be measuring registration
error rather than anatomy.

**The contrasts are not identical, and that is a real limitation rather than a
technicality.** gong2015 contrasts expert with non-expert action gamers -- both
play. Julia2018 contrasts players with non-players. Transfer between them
assumes both index the same underlying dimension of action-game exposure. If
transfer fails, that assumption is one of the candidate explanations, alongside
scanner differences and the null hypothesis that there is nothing to find.

Three things are reported, and the third is the one that matters:

  within      cross-validated accuracy inside each dataset, for reference
  transfer    train on A, test on all of B -- the external validation
  pooled      both datasets together with site as a covariate, for power

A permutation null accompanies the transfer result: labels are shuffled *within
the training dataset only*, the model refitted, and the held-out dataset
predicted, which is the null for "a model trained on A predicts B".

    uv run python scripts/run_cross_dataset.py --atlas aal
"""

from __future__ import annotations

import argparse
import json
import sys
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

from acnets.multiverse import accuracy_interval  # noqa: E402

OUT = Path('outputs/cross_dataset')
GONG = Path('data/gong2015/derivatives')
JULIA = Path('data/Julia2018/derivatives')


def load(atlas: str) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """{dataset: (features, labels, region names)}, on the shared columns."""
    gong = pd.read_csv(GONG / f'gmv_gong2015_{atlas}.csv')
    julia = pd.read_csv(JULIA / f'gmv_julia2018_{atlas}.csv')

    shared = [c for c in gong.columns
              if c in julia.columns and c not in ('subject', 'total_gm_mm3',
                                                  'brain_mask_cm3')]
    if len(shared) < 10:
        raise SystemExit(f'only {len(shared)} shared columns between datasets')

    # gong2015: expert = 0, non-expert = 1.  Julia2018: AVGP = 0, NVGP = 1.
    # The mapping treats "more action-game exposure" as class 0 in both, which
    # is the assumption transfer rests on and is stated in the docstring.
    gong_y = np.array([0 if str(s).startswith('expert') else 1
                       for s in gong['subject']])
    julia_y = np.array([0 if 'AVGP' in str(s) else 1 for s in julia['subject']])

    return {
        'gong2015': (gong[shared].to_numpy(float), gong_y, shared),
        'julia2018': (julia[shared].to_numpy(float), julia_y, shared),
    }


def model():
    return LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())


def within(x: np.ndarray, y: np.ndarray) -> float:
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        fitted = model().fit(scaler.transform(x[train]), y[train])
        predictions[test] = fitted.predict(scaler.transform(x[test]))
    return float((predictions == y).mean())


def transfer(train_x, train_y, test_x, test_y) -> float:
    """Fit on one dataset, predict the other. Nothing from the target is used.

    Standardisation is fitted on the training dataset and applied to the target,
    which is the honest choice: refitting the scaler on the target would let the
    target's distribution influence its own prediction, and z-scoring the target
    within itself would quietly harmonise the two sites.
    """
    scaler = StandardScaler().fit(train_x)
    fitted = model().fit(scaler.transform(train_x), train_y)
    return float((fitted.predict(scaler.transform(test_x)) == test_y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='aal')
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = load(args.atlas)
    for name, (x, y, _cols) in data.items():
        print(f'  {name:10s} n={len(y):3d}  ({int((y == 0).sum())} high-exposure, '
              f'{int((y == 1).sum())} low)  {x.shape[1]} regions')
    print(f'  shared feature space: {len(data["gong2015"][2])} regions '
          f'({args.atlas})\n')

    # sanity: the two datasets must describe the same anatomy before any
    # transfer result means anything (section 19)
    profile = np.corrcoef(data['gong2015'][0].mean(0),
                          data['julia2018'][0].mean(0))[0, 1]
    print(f'regional profile agreement: r = {profile:+.3f}')
    if profile < 0.5:
        raise SystemExit(
            f'profile correlation {profile:+.3f} is too low for the datasets to '
            f'share a feature space; a transfer result would be meaningless. '
            f'Use a volumetric atlas (section 19).')

    results = {}
    print('\nwithin-dataset (leave-one-out, for reference):')
    for name, (x, y, _) in data.items():
        value = within(x, y)
        interval = accuracy_interval(value * len(y), len(y))
        results[f'within/{name}'] = value
        print(f'  {name:10s} {value:.3f}  [{interval.low:.2f}, {interval.high:.2f}]')

    print('\nEXTERNAL VALIDATION (train on one, predict the other):')
    pairs = [('gong2015', 'julia2018'), ('julia2018', 'gong2015')]
    for source, target in pairs:
        sx, sy, _ = data[source]
        tx, ty, _ = data[target]
        value = transfer(sx, sy, tx, ty)
        interval = accuracy_interval(value * len(ty), len(ty))
        results[f'transfer/{source}->{target}'] = value

        rng = np.random.default_rng(args.seed)
        null = np.array(Parallel(n_jobs=args.n_jobs)(
            delayed(transfer)(sx, rng.permutation(sy), tx, ty)
            for _ in range(args.n_permutations)))
        p = float((1 + (null >= value).sum()) / (len(null) + 1))
        results[f'p/{source}->{target}'] = p
        # store what the null looked like, not only the p it produced: a
        # transfer of 0.500 against a null centred on 0.500 and one against a
        # null centred on 0.400 are different results with the same p.
        results[f'null_median/{source}->{target}'] = float(np.median(null))
        results[f'null_95/{source}->{target}'] = float(np.quantile(null, 0.95))
        print(f'  {source:10s} -> {target:10s} {value:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]   '
              f'null median {np.median(null):.3f}   p = {p:.4f}'
              f'{"   SIGNIFICANT" if p < 0.05 else ""}')

    (OUT / f'cross_dataset_{args.atlas}.json').write_text(json.dumps({
        'atlas': args.atlas,
        'profile_correlation': float(profile),
        'n': {k: len(v[1]) for k, v in data.items()},
        'n_permutations': args.n_permutations,
        'results': results,
        'caveat': 'gong2015 contrasts expert vs non-expert gamers; Julia2018 '
                  'contrasts players vs non-players. Transfer assumes both '
                  'index action-game exposure.',
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
