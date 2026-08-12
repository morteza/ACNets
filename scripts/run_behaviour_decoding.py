"""Can behaviour alone classify AVGP against NVGP? A quick estimate.

Proof of concept, not a pipeline: one feature set (accuracy, RT, inverse
efficiency), a handful of classifiers with a nested grid search, repeated
cross-validation, and one family-wise permutation null over the grid.

**Read it as a positive control, not a discovery.** The contrast is near-circular
-- participants were recruited on gaming habits and better attentional
performance is what that selects for -- so a high number here describes the
design. What makes it worth running is that section 3.5 records this project as
lacking any positive control: nothing says what this machinery reaches at n = 32
when a real effect is present. The behavioural group difference is real and
ordinary in size (accuracy d = +0.74, IES d = -0.76), so whatever this returns is
the reference the connectivity result (0.660 against a critical 0.739) should be
read against.

    uv run python scripts/run_behaviour_decoding.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from acnets.multiverse import accuracy_interval, fixed_splits
from run_brain_behavior import behavioural_performance

warnings.filterwarnings('ignore')
OUT = Path('outputs/behaviour_decoding')

FEATURES = ['accuracy', 'rt', 'ies']


def grids(seed):
    """Four fast linear-ish families. Kept small on purpose: this is an
    estimate, and every extra cell widens the null it has to clear."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    return {
        'logistic': (LogisticRegression(max_iter=2000),
                     {'C': [0.01, 0.1, 1, 10]}),
        'svm-linear': (SVC(kernel='linear'), {'C': [0.01, 0.1, 1, 10]}),
        'svm-rbf': (SVC(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': ['scale', 1]}),
        'lda': (LinearDiscriminantAnalysis(), {'solver': ['svd', 'lsqr']}),
    }


def evaluate(x, y, estimator, space, splits, seed):
    """Accuracy over the folds; scaling and the grid search fit inside each."""
    correct = 0
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for train, test in splits:
        search = GridSearchCV(estimator, space, cv=inner, scoring='accuracy')
        model = make_pipeline(StandardScaler(), search)
        model.fit(x[train], y[train])
        correct += int((model.predict(x[test]) == y[test]).sum())
    return correct / sum(len(t) for _, t in splits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-B', '--n-permutations', type=int, default=100)
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    performance = behavioural_performance()
    y = np.array([0 if 'AVGP' in str(i) else 1 for i in performance.index])
    x = performance[FEATURES].to_numpy(float)
    n = len(y)
    splits = fixed_splits(n, 4, args.n_repeats, random_state=args.seed)
    models = grids(args.seed)

    print(f'n = {n} ({(y == 0).sum()} AVGP, {(y == 1).sum()} NVGP), '
          f'{len(FEATURES)} features, {len(models)} classifiers, '
          f'{len(splits)} folds, B = {args.n_permutations}\n')

    observed = {name: evaluate(x, y, est, space, splits, args.seed)
                for name, (est, space) in models.items()}
    for name, accuracy in sorted(observed.items(), key=lambda kv: -kv[1]):
        interval = accuracy_interval(accuracy * n, n)
        print(f'  {name:12s} {accuracy:.4f}  [{interval.low:.2f}, {interval.high:.2f}]')

    rng = np.random.default_rng(args.seed)
    null_max = np.empty(args.n_permutations)
    for i in range(args.n_permutations):
        permuted = rng.permutation(y)
        null_max[i] = max(evaluate(x, permuted, est, space, splits, args.seed)
                          for est, space in models.values())
        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{args.n_permutations} permutations')

    best = max(observed, key=observed.get)
    critical = float(np.percentile(null_max, 95))
    p = float((1 + np.sum(null_max >= observed[best])) / (args.n_permutations + 1))
    print(f'\nbest {observed[best]:.4f} ({best})   critical {critical:.4f}   '
          f'null median {np.median(null_max):.4f}   p = {p:.4f}')
    print('connectivity, same machinery: 0.660 vs critical 0.739, p = 0.291')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'behaviour_decoding.json').write_text(json.dumps({
        'n': int(n), 'features': FEATURES, 'observed': observed,
        'best': best, 'best_accuracy': observed[best],
        'critical_value_95': critical, 'null_max_median': float(np.median(null_max)),
        'p_value': p, 'n_permutations': args.n_permutations,
        'note': 'near-circular; a positive control for the pipeline at n = 32, '
                'not a discovery about the brain',
    }, indent=2))
    print(f'wrote {OUT / "behaviour_decoding.json"}')


if __name__ == '__main__':
    main()
