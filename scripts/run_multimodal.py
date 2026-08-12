"""How high can accuracy go if the pipeline is allowed more than resting connectivity?

The target is 80%. Sections 1-8 of `docs/paper/decodability_ceiling.md` establish
that resting connectivity alone cannot reach it at n = 32 -- the bias-corrected
Mahalanobis distance over the 21 network edges is *negative*, so the data contain
no detectable multivariate separation in that space, and 279 configurations plus
13 tuned classifier families all landed near 0.6.

So this changes the input rather than the model. Four modalities are available
per participant, and only the first has ever been used:

  connectivity  21 network edges (the paper's feature space)
  structural    267 regional grey-matter volumes from fmriprep aparc+aseg
  motion        20 summaries of head movement -- decodes at 0.61 on its own
  behaviour     accuracy, median RT and inverse efficiency on the attention task

**What each of these means is not the same, and the difference matters more than
the accuracy.** Connectivity and structure are measurements of the brain.
Motion is a nuisance variable that happens to differ between the groups.
Behaviour is the *trained skill itself* -- action gamers are faster and more
accurate on attention tasks, which is the premise of the whole literature, so
predicting group membership from it is close to circular. A high number obtained
by adding behaviour is a claim about the design, not a discovery about the brain,
and the report below labels every row accordingly.

Every combination is evaluated on the same label-independent folds used
everywhere else, with all fitting inside training folds, and single-modality
rows reported alongside so a fusion gain can be attributed. Participants missing
a modality are dropped for the combinations that need it, and each row reports
its own n -- accuracies computed on different subsets are not comparable, which
is precisely why the n is printed next to every one.

    uv run python scripts/run_multimodal.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

# xgboost before torch; see acnets._openmp
from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure  # noqa: E402
from acnets.multiverse import accuracy_interval, fixed_splits  # noqa: E402
from acnets.multiverse.classifiers import make_classifier  # noqa: E402
from acnets.multiverse.runner import vectorize  # noqa: E402
from run_agreement import aggregate, load_atlas  # noqa: E402
from run_brain_behavior import behavioural_performance  # noqa: E402
from run_motion_control import load as load_motion  # noqa: E402

OUT = Path('outputs/multimodal')
STATISTIC = 'partial correlation'
CLASSIFIERS = ('svm', 'ridge', 'lda', 'logistic', 'gnb', 'random-forest',
               'xgboost', 'svm-rbf')

# what each modality licenses as a conclusion, carried into the output so the
# distinction survives being copied into a slide
KIND = {
    'connectivity': 'brain measurement',
    'structural': 'brain measurement',
    'motion': 'nuisance variable that differs by group',
    'behaviour': 'the trained skill itself -- near-circular for this contrast',
}


def modalities(seed: int) -> dict[str, pd.DataFrame]:
    dataset = load_atlas('dosenbach2010')
    timeseries, _ = aggregate(dataset, seed)
    subjects = [str(s) for s in dataset['subject'].values]

    measure = ExtraConnectivityMeasure(kind=STATISTIC, vectorize=False)
    measure.fit(timeseries['network'])
    connectivity = pd.DataFrame(
        vectorize(measure.transform(timeseries['network']), False), index=subjects)

    structural = pd.read_csv('outputs/structural_volumes.csv', index_col=0)
    structural.index = [str(i) for i in structural.index]

    motion_table, _ = load_motion()
    motion_table.index = [str(i) for i in motion_table.index]

    behaviour = behavioural_performance()
    behaviour.index = [str(i) for i in behaviour.index]

    return {'connectivity': connectivity, 'structural': structural,
            'motion': motion_table, 'behaviour': behaviour}


def labels_for(index) -> np.ndarray:
    return np.array([0 if str(s).startswith('AVGP') else 1 for s in index])


def score(x: np.ndarray, y: np.ndarray, name: str, seed: int,
          n_splits: int, n_repeats: int) -> float:
    splits = fixed_splits(len(y), n_splits, n_repeats, seed)
    folds = []
    for train, test in splits:
        scaler = StandardScaler().fit(x[train])
        model = make_classifier(name, random_state=seed)
        try:
            model.fit(scaler.transform(x[train]), y[train])
            predicted = model.predict(scaler.transform(x[test]))
        except Exception:
            return float('nan')
        folds.append(float((predicted == y[test]).mean()))
    return float(np.mean(folds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=25)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tables = modalities(args.seed)
    print('modalities:')
    for name, frame in tables.items():
        print(f'  {name:14s} {frame.shape[0]:2d} participants x '
              f'{frame.shape[1]:3d} features   ({KIND[name]})')

    names = list(tables)
    combos = [c for size in range(1, len(names) + 1)
              for c in combinations(names, size)]

    jobs = []
    for combo in combos:
        joined = pd.concat([tables[m] for m in combo], axis=1, join='inner')
        joined = joined.dropna()
        if len(joined) < 16:
            continue
        y = labels_for(joined.index)
        if len(np.unique(y)) < 2:
            continue
        x = joined.to_numpy(dtype=float)
        for classifier in CLASSIFIERS:
            jobs.append((combo, classifier, x, y))

    print(f'\n{len(jobs)} (modality combination x classifier) cells\n')
    values = Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(score)(x, y, classifier, args.seed, args.n_splits, args.n_repeats)
        for _, classifier, x, y in jobs)

    rows = []
    for (combo, classifier, x, y), value in zip(jobs, values, strict=True):
        if not np.isfinite(value):
            continue
        interval = accuracy_interval(value * len(y), len(y))
        rows.append({'modalities': '+'.join(combo), 'classifier': classifier,
                     'accuracy': value, 'n': len(y), 'n_features': x.shape[1],
                     'wilson_low': interval.low, 'wilson_high': interval.high,
                     'brain_only': all(KIND[m] == 'brain measurement'
                                       for m in combo),
                     'uses_behaviour': 'behaviour' in combo})
    table = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    table.to_csv(OUT / 'multimodal.csv', index=False)

    print('\nbest per modality combination:')
    for combo, group in table.groupby('modalities', sort=False):
        best = group.loc[group['accuracy'].idxmax()]
        flag = 'BRAIN ONLY' if best['brain_only'] else (
            'uses behaviour' if best['uses_behaviour'] else 'uses motion')
        print(f'  {best["accuracy"]:.3f}  [{best["wilson_low"]:.2f}, '
              f'{best["wilson_high"]:.2f}]  n={int(best["n"]):2d}  '
              f'{combo:42s} {best["classifier"]:14s} {flag}')

    print('\ntop 10 overall:')
    for _, r in table.head(10).iterrows():
        print(f'  {r["accuracy"]:.3f}  n={int(r["n"]):2d}  '
              f'{r["modalities"]:42s} {r["classifier"]}')

    brain = table[table['brain_only']]
    best_overall = table.iloc[0]
    best_brain = brain.iloc[0] if len(brain) else None
    print(f'\nbest overall            {best_overall["accuracy"]:.3f}  '
          f'({best_overall["modalities"]}, n={int(best_overall["n"])})')
    if best_brain is not None:
        print(f'best brain-measurement  {best_brain["accuracy"]:.3f}  '
              f'({best_brain["modalities"]}, n={int(best_brain["n"])})')
    print(f'reached 80%? {"YES" if best_overall["accuracy"] >= 0.80 else "no"}'
          f' overall; '
          f'{"YES" if best_brain is not None and best_brain["accuracy"] >= 0.80 else "no"}'
          f' from brain measurements alone')

    (OUT / 'multimodal.json').write_text(json.dumps({
        'cv': {'n_splits': args.n_splits, 'n_repeats': args.n_repeats},
        'modality_kinds': KIND,
        'best_overall': best_overall.to_dict(),
        'best_brain_only': best_brain.to_dict() if best_brain is not None else None,
        'note': 'accuracies on different participant subsets are not directly '
                'comparable; every row carries its own n. Rows using behaviour '
                'predict group membership from the trained skill that defines '
                'the groups and are near-circular.',
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
