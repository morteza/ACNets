"""How to get any accuracy you want, and what each step costs in validity.

**Every number this script prints above the first row is invalid.** It exists to
quantify malpractice, not to perform it: each rung applies one widely-seen
methodological shortcut, and reports what that shortcut is worth in accuracy on
this dataset. The last rung reaches 100%.

The point is that the ladder is climbable entirely with defensible-sounding
steps. Nobody sets out to report a fraudulent number. They cross-validate, but
select features first on all the data. They try a few random seeds and report
the best. They tune a scaler outside the fold. Each step feels like a detail;
together they are worth more than the entire effect the study is looking for.

Rungs, in the order they are usually taken:

  0  honest        nested CV, everything fitted inside training folds
  1  best cell     the maximum over a search, reported without its null
  2  selection     features chosen on all participants, then cross-validated
  3  best seed     the best of several fold partitions
  4  leakage       scaler fitted on all data before splitting
  5  no holdout    accuracy on the training data itself

Read alongside `docs/paper/decodability_ceiling.md`, which reports the honest
analyses. A reader who wants to know why this project insists on permutation
nulls and in-fold fitting can read the size of the numbers below instead of an
argument.

    uv run python scripts/run_overfitting_ladder.py
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
from scipy import stats  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure  # noqa: E402
from acnets.multiverse import fixed_splits  # noqa: E402
from acnets.multiverse.classifiers import make_classifier  # noqa: E402
from acnets.multiverse.runner import vectorize  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402

OUT = Path('outputs/ladder')
CLASSIFIERS = ('svm', 'ridge', 'lda', 'logistic', 'gnb', 'random-forest',
               'xgboost', 'svm-rbf')


def region_features(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Region-level edges: 12,880 features for 32 participants.

    Deliberately the high-dimensional space, because that is where selection on
    all the data buys the most -- with enough candidate features, some will
    separate any labelling by chance.
    """
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, _ = aggregate(dataset, seed)
    measure = ExtraConnectivityMeasure(kind='correlation', vectorize=False)
    measure.fit(timeseries['region'])
    return vectorize(measure.transform(timeseries['region']), False), y


def honest(x, y, splits, name='svm') -> float:
    """Rung 0. Everything fitted inside the training fold."""
    folds = []
    for train, test in splits:
        scaler = StandardScaler().fit(x[train])
        a, b = x[train][y[train] == 0], x[train][y[train] == 1]
        score = np.nan_to_num(np.abs(stats.ttest_ind(a, b, equal_var=False).statistic))
        columns = np.argsort(score)[::-1][:10]
        model = make_classifier(name, random_state=0)
        model.fit(scaler.transform(x[train])[:, columns], y[train])
        folds.append((model.predict(scaler.transform(x[test])[:, columns])
                      == y[test]).mean())
    return float(np.mean(folds))


def selection_on_everything(x, y, splits, k=10, name='svm') -> float:
    """Rung 2. Features chosen once, on all participants, then cross-validated.

    This is the single most common way a small-sample paper reports an
    impossible accuracy, and it is easy to write by accident: the selection sits
    above the cross-validation loop in the script, so it looks like part of
    preprocessing rather than part of the model.
    """
    a, b = x[y == 0], x[y == 1]
    score = np.nan_to_num(np.abs(stats.ttest_ind(a, b, equal_var=False).statistic))
    columns = np.argsort(score)[::-1][:k]
    reduced = x[:, columns]

    folds = []
    for train, test in splits:
        scaler = StandardScaler().fit(reduced[train])
        model = make_classifier(name, random_state=0)
        model.fit(scaler.transform(reduced[train]), y[train])
        folds.append((model.predict(scaler.transform(reduced[test])) == y[test]).mean())
    return float(np.mean(folds))


def best_of_seeds(x, y, k=10, n_seeds=20, name='svm') -> tuple[float, float]:
    """Rung 3. The best of several fold partitions. Returns (best, median)."""
    values = []
    for seed in range(n_seeds):
        splits = fixed_splits(len(y), 4, 5, seed)
        values.append(selection_on_everything(x, y, splits, k, name))
    return float(np.max(values)), float(np.median(values))


def leakage(x, y, k=10, name='svm') -> float:
    """Rung 4. Selection and scaling both fitted on everything, then LOOCV."""
    a, b = x[y == 0], x[y == 1]
    score = np.nan_to_num(np.abs(stats.ttest_ind(a, b, equal_var=False).statistic))
    columns = np.argsort(score)[::-1][:k]
    scaled = StandardScaler().fit_transform(x[:, columns])

    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(scaled):
        model = make_classifier(name, random_state=0)
        model.fit(scaled[train], y[train])
        predictions[test] = model.predict(scaled[test])
    return float((predictions == y).mean())


def no_holdout(x, y) -> float:
    """Rung 5. Accuracy on the data the model was fitted to.

    An RBF SVM with a small gamma will memorise 32 points in 12,880 dimensions
    exactly. This is not a weakness of the SVM; it is what unlimited capacity
    against no held-out data always does.
    """
    scaled = StandardScaler().fit_transform(x)
    model = SVC(kernel='rbf', C=1e6, gamma='scale').fit(scaled, y)
    return float((model.predict(scaled) == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    x, y = region_features(args.seed)
    splits = fixed_splits(len(y), 4, 25, args.seed)
    print(f'{len(y)} participants, {x.shape[1]} candidate region-level edges\n')

    rungs = []
    value = honest(x, y, splits)
    rungs.append(('0  honest', value,
                  'selection and scaling inside every training fold'))

    best_cell = max(honest(x, y, splits, c) for c in CLASSIFIERS)
    rungs.append(('1  best cell', best_cell,
                  f'max over {len(CLASSIFIERS)} classifiers, no null reported'))

    value = selection_on_everything(x, y, splits)
    rungs.append(('2  selection on all data', value,
                  'top 10 edges chosen on all 32, then cross-validated'))

    best, median = best_of_seeds(x, y)
    rungs.append(('3  best of 20 seeds', best,
                  f'best fold partition of 20 (median of those 20: {median:.3f})'))

    value = leakage(x, y)
    rungs.append(('4  + scaling leakage, LOOCV', value,
                  'selection and scaling both fitted on everything'))

    value = no_holdout(x, y)
    rungs.append(('5  no held-out data', value,
                  'accuracy on the training set itself'))

    print(f'{"rung":30s} {"accuracy":>9s}   what was done')
    for label, accuracy, description in rungs:
        print(f'{label:30s} {accuracy:9.3f}   {description}')

    honest_value = rungs[0][1]
    print(f'\ntotal manufactured by stacking shortcuts: '
          f'{rungs[-1][1] - honest_value:+.3f} '
          f'({honest_value:.3f} -> {rungs[-1][1]:.3f})')
    print('\nEvery row below the first is invalid. None may be reported as a '
          'result.\nThe honest estimate for this feature space is the first row.')

    (OUT / 'ladder.json').write_text(json.dumps({
        'n_subjects': len(y),
        'n_candidate_features': int(x.shape[1]),
        'rungs': [{'rung': label, 'accuracy': accuracy, 'method': description}
                  for label, accuracy, description in rungs],
        'honest': honest_value,
        'manufactured': rungs[-1][1] - honest_value,
        'WARNING': 'all rows except rung 0 are demonstrations of invalid '
                   'methodology and must never be cited as results',
    }, indent=2, default=float))
    pd.DataFrame(rungs, columns=['rung', 'accuracy', 'method']).to_csv(
        OUT / 'ladder.csv', index=False)
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
