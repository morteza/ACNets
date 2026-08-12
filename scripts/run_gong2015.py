"""gong2015, step 3: the structural arm, through the same machinery.

Expert vs non-expert action gamers (24 vs 23), classified from native-space
regional grey-matter volumes, under exactly the discipline the resting arm
uses: a declared grid of specifications, label-independent fixed folds, and one
family-wise max-statistic permutation null over the whole grid.

This is an *independent sample*, not extra n: different subjects, different
scanner, different modality from Julia2018, so nothing is pooled here (the
pooled analysis is `run_pooled_structural.py`, and it is also null). These are
the T1 scans behind Gong et al. (2015), shared by that paper's authors, so this
is a reanalysis of the closest published precedent.

**Rerun after two defects, and reduced in scope.** The first pass read
skull-contaminated volumes (`ants.get_mask` does not strip skull) over
Dosenbach spheres (unusable for cross-dataset volumetry: profile r = -0.41).
Both are fixed: volumes come from `segment_t1.py` with a warped conservative
MNI brain mask, and the grid runs three *volumetric* atlases. Attribution is
gone from this script rather than recomputed — the accuracy is at chance, and
attribution ranks from a chance-level classifier describe noise. The withdrawn
claim, including the insula check it was built around, is documented in
`docs/paper/decodability_ceiling.md` sections 18-21.

    uv run python scripts/run_gong2015.py --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from acnets.multiverse import (  # noqa: E402
    accuracy_interval,
    fixed_splits,
    max_stat_permutation,
    specification_curve,
)
from acnets.multiverse.runner import _make_classifier  # noqa: E402

DERIVATIVES = Path('data/gong2015/derivatives')
ATLAS_DIR = Path('data/atlases/prepared')
OUT = Path('outputs/gong2015')

# volumetric only. Dosenbach spheres are excluded by construction: a fixed-radius
# sphere in template space samples different tissue in every native brain, which
# is tolerable for connectivity and not for volumetry.
ATLASES = ('aal', 'harvard-oxford', 'schaefer200')
CLASSIFIERS = ('svm', 'xgboost', 'mlp')


def load_features(atlas: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_csv(DERIVATIVES / f'gmv_gong2015_{atlas}.csv')
    labels = pd.read_csv(ATLAS_DIR / f'{atlas}_labels.csv').set_index('label')

    y = np.array([0 if s.startswith('expert') else 1 for s in table['subject']])
    columns = [c for c in table.columns
               if c not in ('subject', 'total_gm_mm3', 'brain_mask_cm3')]
    x = table[columns].to_numpy(dtype=float)
    # name by label id, never by position -- the Schaefer name list carries a
    # 'Background' entry that produced an off-by-one everywhere it was zipped
    # against parcels (Appendix E item 18).
    names = [str(labels.loc[int(c), 'region']) for c in columns]
    return x, y, names


def fold_accuracy(x, y, train, test, classifier, seed) -> float:
    scaler = StandardScaler().fit(x[train])
    model = _make_classifier(classifier, seed)
    model.fit(scaler.transform(x[train]), y[train])
    return float((model.predict(scaler.transform(x[test])) == y[test]).mean())


def run_grid(data, y, splits, seed) -> dict[str, float]:
    """Mean accuracy per (atlas, classifier) specification."""
    accuracies = {}
    for atlas in ATLASES:
        x, _ = data[atlas]
        for classifier in CLASSIFIERS:
            accuracies[f'{atlas}/gmv/{classifier}'] = float(np.mean(
                [fold_accuracy(x, y, train, test, classifier, seed + fold)
                 for fold, (train, test) in enumerate(splits)]))
    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = {}
    y = None
    for atlas in ATLASES:
        x, y, names = load_features(atlas)
        data[atlas] = (x, names)
    n_subjects = len(y)
    print(f'{n_subjects} subjects ({int((y == 0).sum())} expert, '
          f'{int((y == 1).sum())} non-expert); '
          f'{len(ATLASES) * len(CLASSIFIERS)} specifications')

    splits = fixed_splits(n_subjects, 4, 25, args.seed)

    observed = run_grid(data, y, splits, args.seed)
    best = max(observed, key=observed.get)
    print('\nobserved:')
    for key, value in sorted(observed.items(), key=lambda kv: -kv[1]):
        wilson = accuracy_interval(value * n_subjects, n_subjects)
        print(f'  {value:.3f} [{wilson.low:.2f}, {wilson.high:.2f}]  {key}')

    # ---- family-wise null --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    permutations = [rng.permutation(y) for _ in range(args.n_permutations)]
    started = time.perf_counter()
    null_rows = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(run_grid)(data, permuted, splits, args.seed)
        for permuted in permutations)
    null = np.array([[row[k] for k in observed] for row in null_rows])
    print(f'  ({(time.perf_counter() - started) / 60:.1f} min for the null)')

    result = max_stat_permutation(observed, null)
    print(f'\nfamily-wise over {result.n_specifications} specifications, '
          f'B = {result.n_permutations}:')
    print(f'  observed max {result.observed_max:.3f}  ({best})')
    print(f'  critical value (95) {result.critical_value:.3f}')
    print(f'  p = {result.p_value:.4f}')
    curve = pd.DataFrame(specification_curve(observed, result.null_max))
    n_significant = int(curve['significant'].sum())
    print(f'  {n_significant} of {len(curve)} specifications clear the threshold')

    (OUT / 'gong2015.json').write_text(json.dumps({
        'n_subjects': n_subjects,
        'method': ('native-space regional GM volumes; N4 + warped conservative '
                   'MNI brain mask + brain-to-brain affine registration + '
                   'Atropos three-tissue; volumes normalised by total GM; '
                   'affine-only registration is a declared limitation'),
        'atlases': list(ATLASES),
        'observed': observed,
        **result.summary(),
        'attribution': ('withdrawn, not recomputed: the first pass ranked '
                        'skull-contaminated volumes over Dosenbach spheres, and '
                        'attribution from a chance-level classifier describes '
                        'noise either way (decodability_ceiling.md 18-21)'),
        'provenance': 'group labels from directory names; these are the T1 '
                      'scans behind Gong et al. 2015, shared by that paper\'s '
                      'authors (confirmed, plan.md D1)',
    }, indent=2, default=float))
    curve.to_csv(OUT / 'specification_curve.csv', index=False)
    np.save(OUT / 'null_max.npy', result.null_max)
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
