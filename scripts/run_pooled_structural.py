"""Pool gong2015 and Julia2018 into one structural sample of 79.

Treating gong2015's experts as players and its non-experts as non-players gives
40 players and 39 non-players across two sites -- 2.5x the participants of
either dataset alone, and n is the binding constraint everywhere in this project
(section 11).

**The label assumption, stated plainly.** gong2015's "non-experts" are amateur
action gamers; they play. Julia2018's NVGPs are non-players. Equating them
assumes action-game exposure is a single dimension along which "amateur" and
"non-player" sit close enough to share a class. That is a substantive
assumption, it was made deliberately at the user's instruction, and it weakens
the contrast: if amateurs resemble experts more than they resemble non-players,
pooling adds noise rather than power. Any null here is therefore ambiguous
between "no effect" and "the classes were wrongly merged".

**Site is not confounded with group, which is what makes pooling defensible at
all.** gong2015 is 24/23 and Julia2018 is 16/16, so each site is internally
balanced and a classifier cannot use site as a proxy for the label. What site
*does* do is add between-participant variance that can swamp a small group
effect -- the two datasets' absolute volumes differ by roughly 40% because one
arrives skull-stripped and the other does not.

Three conditions separate that:

  naive        pooled with no correction, so the site effect is left in
  harmonised   site means and scales removed, **fitted inside training folds
               only** so no test participant informs its own correction
  site-check   how well the same model predicts *site* rather than group --
               a diagnostic, and it should be near-perfect, which is precisely
               why the harmonisation matters

    uv run python scripts/run_pooled_structural.py --atlas aal
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

OUT = Path('outputs/pooled_structural')
GONG = Path('data/gong2015/derivatives')
JULIA = Path('data/Julia2018/derivatives')


def load(atlas: str):
    """(features, label, site) for the pooled sample.

    label 0 = player (gong expert, Julia AVGP), 1 = non-player.
    """
    gong = pd.read_csv(GONG / f'gmv_gong2015_{atlas}.csv')
    julia = pd.read_csv(JULIA / f'gmv_julia2018_{atlas}.csv')
    shared = [c for c in gong.columns
              if c in julia.columns
              and c not in ('subject', 'total_gm_mm3', 'brain_mask_cm3')]

    x = np.vstack([gong[shared].to_numpy(float), julia[shared].to_numpy(float)])
    y = np.concatenate([
        [0 if str(s).startswith('expert') else 1 for s in gong['subject']],
        [0 if 'AVGP' in str(s) else 1 for s in julia['subject']]])
    site = np.concatenate([np.zeros(len(gong), int), np.ones(len(julia), int)])
    return x, y, site, shared


def harmonise(x_train, site_train, x_apply, site_apply):
    """Remove per-site mean and scale, estimated on the training rows only.

    Fitting the site statistics on all participants would let a held-out
    participant contribute to its own correction. Estimating them per fold costs
    nothing and keeps the estimate honest.
    """
    out = x_apply.copy()
    for value in np.unique(site_apply):
        rows = site_train == value
        if rows.sum() < 2:
            continue
        centre = x_train[rows].mean(axis=0)
        spread = x_train[rows].std(axis=0)
        spread[spread < 1e-12] = 1.0
        target = site_apply == value
        out[target] = (x_apply[target] - centre) / spread
    return out


def model():
    return LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())


def loocv(x, y, site=None) -> float:
    """LOOCV; when `site` is given, harmonise inside each fold."""
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        if site is None:
            scaler = StandardScaler().fit(x[train])
            x_train, x_test = scaler.transform(x[train]), scaler.transform(x[test])
        else:
            x_train = harmonise(x[train], site[train], x[train], site[train])
            x_test = harmonise(x[train], site[train], x[test], site[test])
        fitted = model().fit(x_train, y[train])
        predictions[test] = fitted.predict(x_test)
    return float((predictions == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='aal')
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    x, y, site, _shared = load(args.atlas)
    print(f'pooled n={len(y)}  ({int((y == 0).sum())} players, '
          f'{int((y == 1).sum())} non-players)  {x.shape[1]} regions ({args.atlas})')
    for value, name in ((0, 'gong2015'), (1, 'julia2018')):
        rows = site == value
        print(f'  {name:10s} n={rows.sum():3d}  '
              f'{int((y[rows] == 0).sum())} player / '
              f'{int((y[rows] == 1).sum())} non-player')

    site_accuracy = loocv(x, site)
    print(f'\nsite is predictable at {site_accuracy:.3f} '
          f'-- the effect harmonisation has to remove')

    results = {'site_predictability': site_accuracy}
    print('\ngroup classification:')
    for label, kwargs in (('naive (site left in)', {}),
                          ('harmonised in-fold', {'site': site})):
        value = loocv(x, y, **kwargs)
        interval = accuracy_interval(value * len(y), len(y))
        results[label] = value
        print(f'  {label:24s} {value:.3f}  '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

    best_label = max(('naive (site left in)', 'harmonised in-fold'),
                     key=lambda k: results[k])
    best = results[best_label]
    use_site = site if 'harmon' in best_label else None
    rng = np.random.default_rng(args.seed)
    null = np.array(Parallel(n_jobs=args.n_jobs)(
        delayed(loocv)(x, rng.permutation(y), use_site)
        for _ in range(args.n_permutations)))
    p = float((1 + (null >= best).sum()) / (len(null) + 1))
    print(f'\npermutation null on the better condition ({best_label}), '
          f'B = {args.n_permutations}:')
    print(f'  observed {best:.3f}   null median {np.median(null):.3f}   '
          f'95th pct {np.percentile(null, 95):.3f}   p = {p:.4f}'
          f'{"   SIGNIFICANT" if p < 0.05 else ""}')

    results['p'] = p
    results['null_median'] = float(np.median(null))
    (OUT / f'pooled_{args.atlas}.json').write_text(json.dumps({
        'atlas': args.atlas, 'n': len(y),
        'n_players': int((y == 0).sum()), 'n_nonplayers': int((y == 1).sum()),
        'n_regions': int(x.shape[1]),
        'results': results,
        'label_assumption': "gong2015 non-experts (amateur action gamers) are "
                            "treated as non-players, equating them with "
                            "Julia2018 NVGPs. If amateurs resemble experts, "
                            "pooling adds noise and a null is ambiguous.",
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
