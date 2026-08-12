"""External validation between the two samples, drawn as what it is: a null.

Six tests -- three volumetric atlases by two directions -- each an accuracy
against its own permutation null. The figure exists to make one thing hard to
misread: every observed point sits inside the band its null occupies, so the
absence of an effect is visible rather than asserted.

The second panel carries the check that licenses the first. Transfer between
datasets is only interpretable if the two feature spaces correspond, so the
mean regional volume profiles are correlated across samples and that
correlation is drawn alongside. A near-zero or negative profile correlation --
which is exactly what the Dosenbach sphere atlas gives -- would mean the null
is uninformative rather than negative, and the reader should be able to rule
that out by looking.

    from acnets.plotting.cross_dataset import cross_dataset_figure
    cross_dataset_figure('outputs/cross_dataset').savefig('figureS5.png')
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DIRECTIONS = ('gong2015->julia2018', 'julia2018->gong2015')


def _load(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob('cross_dataset_*.json'))
    if not paths:
        raise FileNotFoundError(
            f'no cross_dataset_*.json in {results_dir}; run '
            f'scripts/run_cross_dataset.py --atlas <name> first')
    return [json.loads(path.read_text()) for path in paths]


def cross_dataset_figure(results_dir: str | Path,
                         figsize: tuple[float, float] = (7.5, 4.2)):
    """Figure: transfer accuracy against its null, per atlas and direction."""
    payloads = _load(Path(results_dir))
    atlases = [p['atlas'] for p in payloads]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=figsize, layout='constrained',
        gridspec_kw={'width_ratios': [3, 1]})

    y, ticks, labels = 0.0, [], []
    for payload in payloads:
        for direction in DIRECTIONS:
            results = payload['results']
            observed = results[f'transfer/{direction}']
            # older stores predate these keys; drawing the point without its
            # null would show an accuracy with nothing to judge it against
            missing = [k for k in ('null_median', 'null_95')
                       if f'{k}/{direction}' not in results]
            if missing:
                raise KeyError(
                    f'{payload["atlas"]} lacks {missing} for {direction}; '
                    f'rerun scripts/run_cross_dataset.py --atlas '
                    f'{payload["atlas"]}')
            median = results[f'null_median/{direction}']
            upper = results[f'null_95/{direction}']

            left.plot([median, upper], [y, y], color='0.75', lw=6,
                      solid_capstyle='butt', zorder=2)
            left.scatter([median], [y], marker='|', s=90, color='0.45',
                         zorder=3)
            left.scatter([observed], [y], s=46, color='0.15', zorder=4)
            left.annotate(f'p = {results[f"p/{direction}"]:.2f}',
                          xy=(0.995, y), xycoords=('axes fraction', 'data'),
                          ha='right', va='center', fontsize=7, color='0.45')
            ticks.append(y)
            labels.append(f'{payload["atlas"]}\n{direction.replace("->", " → ")}')
            y += 1
        y += 0.4

    left.axvline(0.5, color='0.85', lw=1, ls=':', zorder=1)
    left.set_yticks(ticks)
    left.set_yticklabels(labels, fontsize=7)
    left.invert_yaxis()
    left.set_xlabel('transfer accuracy')
    left.set_xlim(0.40, 0.66)
    left.set_title('observed (dot) against its permutation null '
                   '(bar: median to 95th)', fontsize=9, loc='left')
    left.spines[['top', 'right', 'left']].set_visible(False)

    profiles = [p['profile_correlation'] for p in payloads]
    order = np.arange(len(profiles))
    right.barh(order, profiles, color='0.35', height=0.55)
    right.axvline(0.5, color='crimson', lw=1, ls='--')
    right.set_yticks(order)
    right.set_yticklabels(atlases, fontsize=7)
    right.invert_yaxis()
    right.set_xlim(0, 1)
    right.set_xlabel('profile r\n(dashed: validity floor)')
    right.set_title('feature spaces\ncorrespond', fontsize=9, loc='left')
    right.spines[['top', 'right', 'left']].set_visible(False)

    return figure
