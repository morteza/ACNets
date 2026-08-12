"""What the stacked model leans on, and what that is and is not evidence of.

Companion to `run_stacked_all.py`. It answers the "which modality mattered"
question in the only form this design still supports after section 3.4.3 of the
manuscript, which is a narrow one.

The manuscript's permuted-label null showed that attribution rankings from
classifiers that do not decode are reproducible under shuffled labels, because
models fitted to one small sample overfit it in correlated ways. That result
applies with full force here: the stacked model is at chance, so a feature
ranking taken from inside its base learners would describe noise, and would look
just as stable if the labels were destroyed. **No regional or edge-level
attribution is computed here, deliberately.**

What survives is one level up. The meta-learner's coefficients say how much
weight the stack put on each modality *path*, and its availability indicators say
how much it read the missingness pattern itself. Those are facts about the design
-- which measurements were informative enough to be worth combining, and whether
the combiner leaned on who-has-what -- rather than facts about the brain, and
they stay interpretable at chance accuracy because they describe the procedure.

Out-of-fold path scores are also correlated with each other and with the label,
which says whether the paths are redundant or complementary. A design worth
scaling up would show complementary paths; redundant ones mean extra modalities
are buying nothing.

Usage:
    uv run python scripts/run_stacked_weights.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stacked_all import (
    STRUCT_ATLASES,
    available,
    base_scores,
    load_everything,
    meta_features,
)

OUT = Path('outputs/stacked_all')
MODALITIES = ('structural', 'functional', 'behaviour')


def out_of_fold_scores(blocks, y, site, modalities, seed, n_splits=4):
    """One out-of-fold score per participant per modality, NaN where absent."""
    scores = {m: np.full(len(y), np.nan) for m in modalities}
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    index = np.arange(len(y))
    for train, test in folds.split(index, y):
        for modality in modalities:
            scores[modality][test] = base_scores(
                modality, blocks[modality], y, site, train, test, seed)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='schaefer200', choices=STRUCT_ATLASES)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    blocks, y, site, _subjects, _names = load_everything(args.atlas, args.seed)
    n = len(y)
    print(f'pooled n = {n}, atlas = {args.atlas}')
    print('availability: ' + ', '.join(
        f'{m} {int(available(blocks[m]).sum())}/{n}' for m in MODALITIES))

    scores = out_of_fold_scores(blocks, y, site, MODALITIES, args.seed)

    print('\npath discriminability, out of fold '
          '(point-biserial r with the label, on whoever has the modality)')
    per_path = {}
    for modality in MODALITIES:
        value = scores[modality]
        rows = ~np.isnan(value)
        # sign convention: y = 0 is player, so a negative r means higher score
        # for non-players; magnitude is what matters here
        r = float(np.corrcoef(value[rows], y[rows])[0, 1])
        accuracy = float(((value[rows] > 0).astype(int) == y[rows]).mean())
        per_path[modality] = {'n': int(rows.sum()), 'r_with_label': r,
                              'sign_accuracy': accuracy}
        print(f'  {modality:12s} n = {int(rows.sum()):3d}   r = {r:+.3f}   '
              f'accuracy from sign alone = {accuracy:.3f}')

    print('\npath redundancy (correlation between path scores, '
          'on participants having both)')
    redundancy = {}
    for i, left in enumerate(MODALITIES):
        for right in MODALITIES[i + 1:]:
            rows = ~np.isnan(scores[left]) & ~np.isnan(scores[right])
            if rows.sum() < 5:
                print(f'  {left} x {right}: only {int(rows.sum())} shared, skipped')
                continue
            r = float(np.corrcoef(scores[left][rows], scores[right][rows])[0, 1])
            redundancy[f'{left} x {right}'] = {'n': int(rows.sum()), 'r': r}
            print(f'  {left:12s} x {right:12s} n = {int(rows.sum()):3d}   r = {r:+.3f}')

    x = meta_features(scores, MODALITIES)
    columns = []
    for modality in MODALITIES:
        columns.append(f'{modality}:score')
        if not available(blocks[modality]).all():
            columns.append(f'{modality}:available')
    meta = LogisticRegression(C=1.0, max_iter=2000,
                              random_state=args.seed).fit(x, y)

    weights = pd.Series(meta.coef_.ravel(), index=columns)
    print('\nmeta-learner weights, fitted on all 79 '
          '(descriptive: this fit is not cross-validated)')
    for name, value in weights.sort_values(key=abs, ascending=False).items():
        print(f'  {name:24s} {value:+.3f}')

    report = {
        'atlas': args.atlas, 'n': n,
        'availability': {m: int(available(blocks[m]).sum()) for m in MODALITIES},
        'path_discriminability': per_path,
        'path_redundancy': redundancy,
        'meta_weights': weights.to_dict(),
        'interpretation_limit': (
            'No regional or edge-level attribution is reported. The stacked '
            'model is at chance, and manuscript section 3.4.3 shows attribution '
            'rankings from non-decoding models are reproducible under permuted '
            'labels; a feature ranking here would describe correlated '
            'overfitting. The quantities above describe the design -- which '
            'paths carry signal and whether they are redundant -- not the brain.'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stacked_weights.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {OUT / "stacked_weights.json"}')


if __name__ == '__main__':
    main()
