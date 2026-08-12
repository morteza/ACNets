"""Leave-one-out multimodal search, with the family-wise null it requires.

Leave-one-out cross-validation gives each fold 31 training participants instead
of the 24 that 4-fold gives, and on this data that is worth a lot: connectivity
alone goes from 0.666 to 0.750. Adding modalities takes the best cell to 0.840.

Neither number means anything on its own. Both are the maximum over a search of
32 cells (4 modality combinations x 8 classifiers) on the same participants, and
the whole point of this project is that such a maximum has to be compared
against what the same search produces from permuted labels. LOOCV in particular
is a high-variance estimator at n = 25 -- each fold's accuracy is 0 or 1 -- so
the maximum over a search of it is exactly the statistic most likely to look
impressive by accident.

Labels are permuted outside everything, the same search is re-run in full, and
the maximum is recorded. Identical estimator, identical search, identical folds.

    uv run python scripts/run_loocv_null.py --n-permutations 200
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
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse import accuracy_interval, max_stat_permutation  # noqa: E402
from acnets.multiverse.classifiers import make_classifier  # noqa: E402
from run_multimodal import KIND, labels_for, modalities  # noqa: E402

OUT = Path('outputs/loocv')
COMBOS = {
    'connectivity': ['connectivity'],
    'behaviour': ['behaviour'],
    'connectivity+behaviour': ['connectivity', 'behaviour'],
    'connectivity+motion+behaviour': ['connectivity', 'motion', 'behaviour'],
}
CLASSIFIERS = ('svm', 'ridge', 'lda', 'logistic', 'gnb', 'random-forest',
               'xgboost', 'svm-rbf')


def loocv(x: np.ndarray, y: np.ndarray, name: str, seed: int = 0) -> float:
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(x):
        scaler = StandardScaler().fit(x[train])
        model = make_classifier(name, random_state=seed)
        try:
            model.fit(scaler.transform(x[train]), y[train])
            predictions[test] = model.predict(scaler.transform(x[test]))
        except Exception:
            return float('nan')
    return float((predictions == y).mean())


def search(blocks: dict[str, tuple[np.ndarray, np.ndarray]],
           permute: np.random.Generator | None = None) -> dict[str, float]:
    """Every (combination, classifier) cell, optionally on permuted labels."""
    out = {}
    for label, (x, y) in blocks.items():
        target = permute.permutation(y) if permute is not None else y
        for classifier in CLASSIFIERS:
            out[f'{label}/{classifier}'] = loocv(x, target, classifier)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tables = modalities(args.seed)

    blocks = {}
    for label, names in COMBOS.items():
        joined = pd.concat([tables[m] for m in names], axis=1,
                           join='inner').dropna()
        blocks[label] = (joined.to_numpy(dtype=float), labels_for(joined.index))
        print(f'  {label:32s} n={len(joined):2d}  '
              f'{joined.shape[1]:3d} features')

    observed = {k: v for k, v in search(blocks).items() if np.isfinite(v)}
    best = max(observed, key=observed.get)
    n_best = len(blocks[best.split('/')[0]][1])
    interval = accuracy_interval(observed[best] * n_best, n_best)
    print(f'\nobserved best {observed[best]:.3f} ({best}, n={n_best}) '
          f'Wilson [{interval.low:.2f}, {interval.high:.2f}]')

    started = time.perf_counter()
    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}:')

    def one(i: int) -> list[float]:
        rng = np.random.default_rng(args.seed + 1 + i)
        scores = search(blocks, permute=rng)
        return [scores[k] for k in observed]

    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(one)(i) for i in range(args.n_permutations)))
    print(f'  {time.perf_counter() - started:.0f}s')

    result = max_stat_permutation(observed, null)
    print(f'\n  observed max        {result.observed_max:.3f}')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  null max, 95th pct  {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')
    n_significant = sum(v >= result.critical_value for v in observed.values())
    print(f'  {n_significant} of {len(observed)} cells clear the critical value')

    print('\nper cell, best classifier per combination:')
    for label in COMBOS:
        cells = {k: v for k, v in observed.items() if k.startswith(f'{label}/')}
        top = max(cells, key=cells.get)
        print(f'  {label:32s} {cells[top]:.3f}  {top.split("/")[1]:14s} '
              f'{"clears" if cells[top] >= result.critical_value else "does not clear"}')

    (OUT / 'loocv.json').write_text(json.dumps({
        'estimator': 'leave-one-out',
        'modality_kinds': KIND,
        'n_cells': len(observed),
        'observed': observed,
        **result.summary(),
        'n_significant': int(n_significant),
        'note': 'LOOCV accuracy at n = 25 is a high-variance statistic and this '
                'is its maximum over a 32-cell search; the null is what makes it '
                'interpretable. Cells using behaviour predict group membership '
                'from the trained skill that defines the groups.',
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
