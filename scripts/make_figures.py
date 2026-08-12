"""Render every manuscript figure from what is already in `outputs/`.

Figures are projections over the stored results, never a second computation, so
this script fits nothing and takes seconds. Rerun it after any analysis rerun.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from acnets.plotting import specification_curve_figure
from acnets.plotting.agreement_figures import (
    agreement_vs_threshold_figure,
    selection_frequency_figure,
    stability_comparison_figure,
)
from acnets.plotting.cross_dataset import cross_dataset_figure

OUT = Path('outputs/figures')

FIGURES = (
    ('figure2_specification_curve',
     lambda: specification_curve_figure('outputs/permutation')),
    ('figure3_agreement_vs_threshold',
     lambda: agreement_vs_threshold_figure('outputs/agreement')),
    ('figure4_stability_comparison',
     lambda: stability_comparison_figure('outputs/agreement')),
    ('figure5_selection_frequency',
     lambda: selection_frequency_figure('outputs/agreement')),
    ('figureS7_gong2015_specification_curve',
     lambda: specification_curve_figure(
         'outputs/gong2015',
         title='gong2015 structural arm: 3 volumetric atlases x 3 classifiers')),
    ('figureS8_cross_dataset_validation',
     lambda: cross_dataset_figure('outputs/cross_dataset')),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, build in FIGURES:
        try:
            figure = build()
        except (FileNotFoundError, KeyError, ValueError) as error:
            # a missing input is a normal state (that analysis has not been
            # rerun yet); it should be named, not swallowed, and not fatal
            failures.append((name, f'{type(error).__name__}: {error}'))
            print(f'  skipped {name}: {type(error).__name__}: {error}')
            continue
        for suffix in ('png', 'pdf'):
            figure.savefig(OUT / f'{name}.{suffix}', dpi=200)
        figure.clf()
        print(f'  wrote {name}.png / .pdf')

    print(f'\n{len(FIGURES) - len(failures)} of {len(FIGURES)} figures written '
          f'to {OUT}/')
    if failures:
        raise SystemExit(
            'missing inputs for: ' + ', '.join(name for name, _ in failures))


if __name__ == '__main__':
    main()
