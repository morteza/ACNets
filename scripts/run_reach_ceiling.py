"""Close the gap between the observed 0.666 and the ~0.72 ceiling, honestly.

Section 11 of `docs/paper/decodability_ceiling.md` puts the ceiling for this
design at roughly 0.72-0.79, set by measurement reliability. The observed best
under 4-fold CV is 0.666. That gap is small but real, and unlike every earlier
attempt in this project there is a principled reason to think it can be closed
rather than searched into existence.

The three changes, each motivated by the diagnosis rather than by trying things:

**Leave-one-out cross-validation.** The learning curve in section 11.3 rises at
+0.0046 accuracy per training participant across the range tested. LOOCV trains
on 31 instead of 24. That is worth roughly +0.03 by the curve's own slope --
predicted in advance, not discovered afterwards.

**Covariance shrinkage chosen analytically, not searched.** The bias-corrected
Mahalanobis distance over 21 edges is negative, which is a statement that the
sample covariance is unusable at this ratio of features to participants. The
remedy is an analytic shrinkage estimator -- Ledoit-Wolf or OAS -- which has a
closed form, needs no tuning, and is designed for exactly this regime. This is
the single most appropriate method available and it had never been tried.

**Ensembling across connectivity statistics rather than choosing among them.**
The multiverse treats the statistics as competing paths and reports the best,
which pays the family-wise penalty. Averaging their decision functions instead
is a variance reduction: the statistics share signal and differ in noise, so the
mean is better conditioned than any member. This converts the multiverse's
liability into an asset.

The specification set is small and fixed before running: 4 statistics x 5
models, plus 5 ensembles, 25 cells. A small family is the point -- the
family-wise critical value grows with the size of the search, so a focused
analysis has real power where a 279-cell sweep has none.

The null uses the identical estimator and the identical set.

    uv run python scripts/run_reach_ceiling.py --n-permutations 500
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
from sklearn.covariance import OAS, LedoitWolf  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeClassifier  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure  # noqa: E402
from acnets.multiverse import accuracy_interval, max_stat_permutation  # noqa: E402
from acnets.multiverse.runner import vectorize  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402

OUT = Path('outputs/ceiling')
STATISTICS = ('correlation', 'partial correlation', 'tangent', 'precision')


def models() -> dict:
    """Five estimators, all strongly regularised, none requiring a search.

    The two shrinkage discriminants are the methodologically correct choice for
    p close to n and are the reason to expect a gain: Ledoit-Wolf and OAS both
    estimate the shrinkage coefficient in closed form from the data, so they
    adapt without spending any of the sample on tuning.
    """
    return {
        'lda-ledoitwolf': lambda: LinearDiscriminantAnalysis(
            solver='lsqr', covariance_estimator=LedoitWolf()),
        'lda-oas': lambda: LinearDiscriminantAnalysis(
            solver='lsqr', covariance_estimator=OAS()),
        'lda-auto': lambda: LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto'),
        'ridge': lambda: RidgeClassifier(alpha=10.0),
        'logistic': lambda: LogisticRegression(C=0.1, max_iter=5000),
        'svm': lambda: LinearSVC(C=0.1, dual='auto', max_iter=20000),
    }


def decision(model, x) -> np.ndarray:
    """A signed score, whichever interface the estimator exposes."""
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(x)).ravel()
    return np.asarray(model.predict_proba(x))[:, 1] - 0.5


def loocv_scores(views: dict[str, np.ndarray], y: np.ndarray,
                 name: str, factory) -> dict[str, float]:
    """LOOCV accuracy per statistic, plus the across-statistic ensemble.

    One pass over the folds computes both, so the ensemble costs nothing extra
    and is guaranteed to use the identical folds as its members.
    """
    per_view = {view: np.empty(len(y)) for view in views}
    ensemble = np.zeros(len(y))

    for train, test in LeaveOneOut().split(y):
        margins = []
        for view, x in views.items():
            scaler = StandardScaler().fit(x[train])
            model = factory()
            model.fit(scaler.transform(x[train]), y[train])
            margin = decision(model, scaler.transform(x[test]))[0]
            per_view[view][test] = margin
            # scale each member's margin to unit training spread before
            # averaging, or whichever statistic has the largest raw margins
            # dominates the ensemble for reasons unrelated to its accuracy
            spread = np.std(decision(model, scaler.transform(x[train])))
            margins.append(margin / (spread if spread > 1e-12 else 1.0))
        ensemble[test] = float(np.mean(margins))

    out = {f'{view}/{name}': float(((margin > 0).astype(int) == y).mean())
           for view, margin in per_view.items()}
    out[f'ensemble/{name}'] = float(((ensemble > 0).astype(int) == y).mean())
    return out


def build_views(seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, _ = aggregate(dataset, seed)
    views = {}
    for statistic in STATISTICS:
        measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
        measure.fit(timeseries['network'])
        views[statistic] = vectorize(measure.transform(timeseries['network']), False)
    return views, y


def run_all(views, y) -> dict[str, float]:
    out = {}
    for name, factory in models().items():
        out.update(loocv_scores(views, y, name, factory))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    views, y = build_views(args.seed)
    print(f'{len(y)} participants, {len(views)} connectivity statistics x '
          f'{len(models())} models + ensembles')
    print('estimator: leave-one-out\n')

    observed = run_all(views, y)
    table = pd.DataFrame(
        [{'cell': k, 'accuracy': v} for k, v in observed.items()]
    ).sort_values('accuracy', ascending=False)

    print(f'{"cell":40s} {"accuracy":>9s}   Wilson 95%')
    for _, row in table.iterrows():
        interval = accuracy_interval(row['accuracy'] * len(y), len(y))
        print(f'  {row["cell"]:38s} {row["accuracy"]:9.3f}   '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

    best = table.iloc[0]
    print(f'\nbest {best["accuracy"]:.3f}  ({best["cell"]})')

    started = time.perf_counter()
    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}, identical estimator:')

    def one(i: int) -> list[float]:
        rng = np.random.default_rng(args.seed + 1 + i)
        scores = run_all(views, rng.permutation(y))
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
    print(f'  {len(clear)} of {len(observed)} cells clear the critical value')
    for k, v in sorted(clear.items(), key=lambda kv: -kv[1]):
        print(f'    {v:.3f}  {k}')

    payload = {'estimator': 'leave-one-out', 'n_subjects': len(y),
               'statistics': list(STATISTICS), 'models': list(models()),
               'observed': observed, **result.summary(),
               'cells_clearing': clear,
               'ceiling_reference': {
                   'reliability_ceiling_at_d2': 0.705,
                   'bayes_ceiling_best_edge': 0.721,
                   'previous_best_4fold': 0.666}}
    (OUT / 'reach.json').write_text(json.dumps(payload, indent=2, default=float))
    table.to_csv(OUT / 'reach.csv', index=False)
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
