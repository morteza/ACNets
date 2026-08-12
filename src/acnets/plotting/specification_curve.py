"""The paper's signature figure: every specification, sorted, against its null.

A specification curve answers the question a multiverse creates for itself. It
plots accuracy for every analysis path, sorted, with a dot matrix underneath
recording which choices each path made, and the family-wise critical value drawn
across the whole thing. A reader can then see at once how much the answer moved,
which choices moved it, and whether any path cleared the bar once the search is
corrected for -- none of which is visible from a single reported number.

The two panels share an x axis by construction: column i of the dot matrix is
specification i in the sorted order above it. That coupling is the entire point,
so the two are drawn by one function rather than composed by the caller.

    from acnets.plotting import specification_curve_figure
    specification_curve_figure('outputs/permutation').savefig('figure2.png')
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# order matters: this is the row order of the dot matrix, coarsest choice first
DIMENSIONS = ('atlas', 'statistic', 'aggregation', 'classifier', 'pretraining')


def parse_labels(spec_ids: pd.Series) -> pd.DataFrame:
    """Recover the choice per dimension from `Specification.label` strings.

    Labels look like `dosenbach2010/pcorr/network/xgboost` with an optional
    `+lemon` suffix. Parsing them back is preferable to threading a second file
    through, and a malformed label raises rather than silently producing an
    empty dot matrix.
    """
    rows = []
    for label in spec_ids:
        text = str(label)
        pretraining = 'none'
        if '+' in text:
            text, pretraining = text.rsplit('+', 1)
        parts = text.split('/')
        if len(parts) == 4:
            atlas, statistic, aggregation, classifier = parts
        elif len(parts) == 3:
            # the structural arm has no aggregation axis: atlas/gmv/classifier
            atlas, statistic, classifier = parts
            aggregation = 'n/a'
        else:
            raise ValueError(f'cannot parse specification label {label!r}: '
                             f'expected atlas/statistic/aggregation/classifier')
        rows.append({'atlas': atlas, 'statistic': statistic,
                     'aggregation': aggregation, 'classifier': classifier,
                     'pretraining': pretraining})
    return pd.DataFrame(rows, index=spec_ids.index)


def _load_summary(results_dir: Path, summary_name: str | None) -> dict:
    if summary_name is not None:
        return json.loads((results_dir / summary_name).read_text())

    candidates = sorted(results_dir.glob('*.json'))
    for path in [results_dir / 'permutation.json', *candidates]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if 'critical_value_95' in payload:
            return payload
    raise FileNotFoundError(
        f'no JSON in {results_dir} carries a critical_value_95; '
        f'looked at {[p.name for p in candidates]}')


def specification_curve_figure(results_dir: str | Path,
                               title: str | None = None,
                               figsize: tuple[float, float] | None = None,
                               summary_name: str | None = None):
    """Figure: sorted accuracy over specifications, with the choice matrix below.

    Reads `specification_curve.csv` and a max-statistic summary JSON from
    `results_dir`. `scripts/run_permutation.py` names the latter
    `permutation.json`; other arms name it after themselves (the gong2015 arm
    writes `gong2015.json`), so when `summary_name` is not given the directory
    is searched for the one JSON carrying a `critical_value_95`.
    """
    results_dir = Path(results_dir)
    curve = pd.read_csv(results_dir / 'specification_curve.csv')
    summary = _load_summary(results_dir, summary_name)

    curve = curve.sort_values('accuracy').reset_index(drop=True)
    choices = parse_labels(curve['spec_id'])

    # A dimension with one level carries no information and would waste a row.
    rows = [(d, sorted(choices[d].unique()))
            for d in DIMENSIONS if choices[d].nunique() > 1]
    n_rows = sum(len(levels) for _, levels in rows)

    n_specs = len(curve)
    if figsize is None:
        figsize = (max(7.0, 0.22 * n_specs), 5.0 + 0.16 * n_rows)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, layout='constrained',
        gridspec_kw={'height_ratios': [3, max(1.2, 0.09 * n_rows)]})
    figure.get_layout_engine().set(hspace=0.02)

    x = np.arange(n_specs)
    # `MaxStatResult.summary()` names this `critical_value_95`. Reading it with
    # a plain .get() and a None fallback silently produced a figure with no
    # threshold line at all, which is the one element the figure exists to show;
    # missing it is an error, not a default.
    if 'critical_value_95' not in summary:
        raise KeyError(f'{results_dir}/permutation.json has no '
                       f'critical_value_95; keys are {sorted(summary)}')
    critical = float(summary['critical_value_95'])
    significant = curve['significant'].to_numpy(dtype=bool)

    top.axhline(0.5, color='0.75', lw=1, ls=':', zorder=1)
    top.axhline(critical, color='crimson', lw=1.4, zorder=6)
    top.annotate(f'family-wise critical value {critical:.3f}',
                 xy=(0.02, critical), xycoords=('axes fraction', 'data'),
                 ha='left', va='bottom', fontsize=8, color='crimson')
    top.vlines(x, 0.5, curve['accuracy'], color='0.8', lw=1, zorder=3)
    top.scatter(x[~significant], curve['accuracy'][~significant],
                s=22, color='0.25', zorder=4, label='not significant')
    if significant.any():
        top.scatter(x[significant], curve['accuracy'][significant],
                    s=30, color='crimson', zorder=5, label='clears the null')
        top.legend(frameon=False, fontsize=8, loc='upper left')

    top.set_ylabel('accuracy')
    top.set_ylim(min(0.45, curve['accuracy'].min() - 0.02),
                 max(0.75, curve['accuracy'].max() + 0.04))
    top.spines[['top', 'right']].set_visible(False)

    if title is None:
        n_significant = int(significant.sum())
        title = (f'{n_specs} specifications, '
                 f'{n_significant} clearing the family-wise null'
                 f' (B = {summary.get("n_permutations", "?")})')
    top.set_title(title, fontsize=10, loc='left')

    # ---- dot matrix -------------------------------------------------------
    labels, y = [], 0
    for dimension, levels in rows:
        for level in levels:
            hit = (choices[dimension] == level).to_numpy()
            bottom.scatter(x[hit], np.full(hit.sum(), y), s=14,
                           color='0.25', marker='|', linewidths=1.6)
            labels.append(f'{level}')
            y += 1
        # a faint rule between dimensions, so the blocks read as blocks
        if y < n_rows:
            bottom.axhline(y - 0.5, color='0.9', lw=0.8)

    bottom.set_yticks(range(n_rows))
    bottom.set_yticklabels(labels, fontsize=7)
    bottom.set_ylim(-0.6, n_rows - 0.4)
    bottom.invert_yaxis()
    bottom.set_xlabel('specifications, sorted by accuracy')
    bottom.set_xlim(-1, n_specs)
    bottom.set_xticks([])
    bottom.spines[['top', 'right', 'bottom']].set_visible(False)

    return figure
