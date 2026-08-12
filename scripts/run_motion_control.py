"""Negative control: can head motion alone tell the groups apart?

Section 3.5 asserts that group differences in connectivity are unlikely to
reflect movement behaviour. That assertion is only worth anything if the
alternative was actually tested, so this fits the same classifiers, on the same
label-independent folds, to fMRIPrep's motion confounds instead of to
connectivity -- and reports the accuracy alongside a per-confound group
comparison.

Two things are checked, and they answer different objections:

1. *Multivariate.* A classifier receiving only motion summaries. If motion
   carried group information, this would decode, and every connectivity result
   would be suspect no matter how confound regression was configured.
2. *Univariate.* Mean framewise displacement, DVARS and RMS displacement per
   group, with Welch t-tests and Cohen's d. A reviewer asks for this table
   whether or not the classifier is at chance, because a mean difference in
   motion is the standard confound in this literature.

Features are per-subject summaries of the six realignment parameters (mean and
SD of each), plus mean and SD of framewise displacement, DVARS and RMS
displacement: 18 features, deliberately the kind of summary a motion-based
account would predict to be discriminative.

    uv run python scripts/run_motion_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from acnets.multiverse import accuracy_interval, fixed_splits
from acnets.multiverse.runner import _make_classifier

FMRIPREP = Path('data/Julia2018/derivatives/fmriprep_2020')
OUT = Path('outputs/controls')

REALIGNMENT = ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z')
SUMMARY = ('framewise_displacement', 'dvars', 'std_dvars', 'rmsd')
CLASSIFIERS = ('svm', 'xgboost', 'mlp')


def subject_motion(path: Path) -> dict[str, float]:
    """Mean and SD of every motion channel, for one participant."""
    frame = pd.read_csv(path, sep='\t')
    features: dict[str, float] = {}
    for column in (*REALIGNMENT, *SUMMARY):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors='coerce').to_numpy()
        values = values[np.isfinite(values)]
        features[f'{column}_mean'] = float(values.mean())
        features[f'{column}_sd'] = float(values.std())
    return features


def load() -> tuple[pd.DataFrame, np.ndarray]:
    rows, subjects = [], []
    for path in sorted(FMRIPREP.glob('sub-*/ses-rest/func/*confounds_timeseries.tsv')):
        subject = path.name.split('_')[0].removeprefix('sub-')
        rows.append(subject_motion(path))
        subjects.append(subject)
    table = pd.DataFrame(rows, index=subjects).sort_index()
    # AVGP vs NVGP, the same convention the connectivity arm uses
    y = np.array([0 if s.startswith('AVGP') else 1 for s in table.index])
    return table, y


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                      + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table, y = load()
    x = table.to_numpy(dtype=float)
    print(f'{len(table)} participants ({int((y == 0).sum())} AVGP, '
          f'{int((y == 1).sum())} NVGP), {x.shape[1]} motion features\n')

    # ---- multivariate: can any classifier decode group from motion? --------
    splits = fixed_splits(len(y), n_splits=4, n_repeats=25, random_state=0)
    results = {}
    for name in CLASSIFIERS:
        folds = []
        for train, test in splits:
            scaler = StandardScaler().fit(x[train])
            model = _make_classifier(name, 0)
            model.fit(scaler.transform(x[train]), y[train])
            folds.append(float((model.predict(scaler.transform(x[test])) == y[test]).mean()))
        accuracy = float(np.mean(folds))
        interval = accuracy_interval(accuracy * len(y), len(y))
        results[name] = {'accuracy': accuracy, 'sd': float(np.std(folds)),
                         'wilson_ci': [interval.low, interval.high]}
        print(f'  {name:8s} {accuracy:.3f}   '
              f'Wilson 95% [{interval.low:.2f}, {interval.high:.2f}]')

    best = max(results, key=lambda k: results[k]['accuracy'])
    print(f'\nbest motion-only classifier: {results[best]["accuracy"]:.3f} ({best})')

    # ---- univariate: does any single motion channel differ by group? ------
    print('\nper-channel group comparison (Welch):')
    comparisons = {}
    for column in table.columns:
        a, b = table[column].to_numpy()[y == 0], table[column].to_numpy()[y == 1]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        comparisons[column] = {'avgp_mean': float(a.mean()), 'nvgp_mean': float(b.mean()),
                               't': float(t), 'p': float(p), 'd': cohens_d(a, b)}
    ordered = sorted(comparisons.items(), key=lambda kv: kv[1]['p'])
    for column, c in ordered[:6]:
        print(f'  {column:36s} AVGP {c["avgp_mean"]:8.4f}  NVGP {c["nvgp_mean"]:8.4f}  '
              f'd = {c["d"]:+.2f}  p = {c["p"]:.3f}')

    n_raw = sum(1 for c in comparisons.values() if c['p'] < 0.05)
    # Holm over every channel summary present; the uncorrected count is reported too because
    # "no channel survives correction" and "no channel is nominally significant"
    # are different claims and a reviewer will want both.
    n_tests = len(comparisons)
    survives_holm = [
        column for rank, (column, c) in enumerate(ordered)
        if c['p'] < 0.05 / (n_tests - rank)]
    print(f'\n{n_raw} of {n_tests} channels nominally significant at 0.05; '
          f'{len(survives_holm)} survive Holm correction')

    payload = {'n_subjects': len(y), 'n_features': int(x.shape[1]),
               'cv': {'n_splits': 4, 'n_repeats': 25},
               'classifiers': results, 'best': best,
               'univariate': comparisons,
               'n_nominally_significant': n_raw,
               'n_surviving_holm': len(survives_holm),
               'surviving_holm': survives_holm}
    (OUT / 'motion.json').write_text(json.dumps(payload, indent=2))
    table.to_csv(OUT / 'motion_features.csv')
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
