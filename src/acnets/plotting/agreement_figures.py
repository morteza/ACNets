"""Figures 3-5: the attribution-agreement side of the argument.

Three small figures, each reading one file that `scripts/run_agreement.py`
already writes. They are separate from the specification curve because they
answer the other half of the question: not "how much did accuracy move" but
"did the explanation move with it".
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def agreement_vs_threshold_figure(results_dir: str | Path, scale: str = 'network'):
    """Figure 3: agreement as a function of the accuracy cutoff.

    The point of plotting the whole curve rather than one number is that a
    rising curve would mean the agreement is a by-product of selecting
    well-performing specifications. The specification count is annotated per
    point because the right-hand end is always built on very few of them.
    """
    results_dir = Path(results_dir)
    curve = pd.read_csv(results_dir / 'agreement_vs_threshold.csv').dropna(
        subset=['agreement'])
    baseline = _baseline(results_dir, scale)

    figure, axes = plt.subplots(figsize=(6.0, 4.0), layout='constrained')
    axes.fill_between(curve['threshold'], curve['ci_low'], curve['ci_high'],
                      color='steelblue', alpha=0.18, lw=0)
    axes.plot(curve['threshold'], curve['agreement'], 'o-', color='steelblue',
              label='observed (95% bootstrap)')
    axes.axhline(baseline, color='crimson', ls='--', lw=1.2,
                 label=f'random rankings ({baseline:.3f})')

    for _, row in curve.iterrows():
        axes.annotate(f'n = {int(row["n_specifications"])}',
                      xy=(row['threshold'], row['agreement']),
                      xytext=(0, 9), textcoords='offset points',
                      ha='center', fontsize=7, color='0.35')

    axes.set_xlabel('accuracy cutoff for including a specification')
    axes.set_ylabel('mean pairwise agreement (RBO)')
    axes.set_ylim(min(baseline, curve['ci_low'].min()) - 0.03,
                  curve['ci_high'].max() + 0.05)
    axes.legend(frameon=False, fontsize=8, loc='lower right')
    axes.spines[['top', 'right']].set_visible(False)
    axes.set_title('Attribution agreement does not depend on the cutoff',
                   fontsize=10, loc='left')
    return figure


def stability_comparison_figure(results_dir: str | Path):
    """Figure 4: attribution and accuracy stability on one normalised scale.

    Both quantities are expressed as agreement in excess of their own empirical
    null, divided by the room above that null, so 0 is chance and 1 is perfect.
    Without that normalisation the two are not comparable at all: raw ranking
    agreement has a floor set by the vocabulary size, and raw accuracy agreement
    has one set by binomial noise.
    """
    results_dir = Path(results_dir)
    payload = json.loads((results_dir / 'agreement.json').read_text())

    rows = []
    for report in payload['reports']:
        comparison = report.get('stability_comparison')
        if not comparison:
            continue
        rows.append({'scale': report['scale'],
                     'attribution': comparison['attribution']['normalized'],
                     'accuracy': comparison['accuracy']['normalized'],
                     'n': report['n_specifications']})
    if not rows:
        raise ValueError(f'no stability_comparison in {results_dir}/agreement.json')

    frame = pd.DataFrame(rows)
    frame['scale'] = (frame['scale']
                      .str.replace('random_network', 'random network')
                      .str.replace(' (symmetric statistics only)',
                                   ', symmetric only', regex=False))
    y = np.arange(len(frame))
    height = 0.38

    figure, axes = plt.subplots(
        figsize=(9.0, 1.6 + 0.8 * len(frame)), layout='constrained')
    axes.barh(y - height / 2, frame['attribution'], height,
              color='steelblue', label='attribution (RBO vs random rankings)')
    axes.barh(y + height / 2, frame['accuracy'], height,
              color='0.6', label='accuracy (vs chance classifiers at n = 32)')
    axes.axvline(0, color='0.3', lw=1)

    axes.set_yticks(y)
    axes.set_yticklabels([f'{r.scale}\n({r.n} specs)' for r in frame.itertuples()],
                         fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel('stability, normalised against its own null '
                    '(0 = chance, 1 = perfect)')
    axes.set_xlim(0, max(0.55, frame[['attribution', 'accuracy']].to_numpy().max() + 0.12))
    axes.legend(frameon=False, fontsize=8, loc='lower right')
    axes.spines[['top', 'right']].set_visible(False)
    axes.set_title('Explanation is more stable than prediction,\n'
                   'and the shuffled control reverses it',
                   fontsize=10, loc='left')
    return figure


def selection_frequency_figure(results_dir: str | Path, top: int = 12):
    """Figure 5: how often each feature lands in the top five."""
    results_dir = Path(results_dir)
    frame = pd.read_csv(results_dir / 'selection_frequency_network.csv',
                        index_col=0)
    column = frame.columns[0]
    frame = frame.sort_values(column, ascending=False).head(top)

    figure, axes = plt.subplots(
        figsize=(6.0, 0.9 + 0.32 * len(frame)), layout='constrained')
    y = np.arange(len(frame))
    colours = ['steelblue' if v >= 0.5 else '0.65' for v in frame[column]]
    axes.barh(y, frame[column], color=colours)
    axes.set_yticks(y)
    axes.set_yticklabels(frame.index, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlim(0, 1.0)
    axes.set_xlabel('fraction of specifications ranking the feature in the top 5')
    axes.spines[['top', 'right']].set_visible(False)
    axes.set_title('Which features the multiverse agrees on', fontsize=10, loc='left')
    return figure


def _baseline(results_dir: Path, scale: str) -> float:
    """Random-ranking floor for `scale`, from the agreement report."""
    payload = json.loads((results_dir / 'agreement.json').read_text())
    for report in payload['reports']:
        if report['scale'] == scale:
            return float(report['rbo']['baseline_mean'])
    raise ValueError(f'no report for scale {scale!r} in {results_dir}')
