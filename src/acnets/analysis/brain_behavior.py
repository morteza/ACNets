"""Relating connectivity features to attentional task performance.

Under a transfer framing this is load-bearing rather than optional. The claim is
that reconfigured control-to-sensorimotor coupling is what supports improvement
across many tasks, and a claim of that shape should bear some relation to
measured performance.

The central methodological point is the distinction between pooled and
within-group association, and the module is built around forcing that
distinction into the open. A pooled correlation across AVGPs and NVGPs can be
produced entirely by the group difference itself: if AVGPs have both higher
coupling and better performance, the two covary across the combined sample even
when coupling explains nothing about performance within either group. This is
Simpson's paradox in its mildest form, and it is the single easiest way to
overstate a brain-behaviour result in a two-group design. Only a within-group
association speaks to whether coupling tracks ability.

Behavioural data come from acnets.preprocessing.beh, which reconstructs trials
with response times from the task session reported in Foecker et al. (2018).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Association:
    """A single brain-behaviour association with a bootstrap interval."""

    label: str
    n: int
    r: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def __str__(self) -> str:
        return (f'{self.label}: r = {self.r:.3f} '
                f'[{self.ci_low:.3f}, {self.ci_high:.3f}], n = {self.n}')


def _bootstrap_r(x: np.ndarray,
                 y: np.ndarray,
                 n_boot: int,
                 random_state: int | None) -> tuple[float, float, float]:
    """Pearson r with a percentile bootstrap interval over participants."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = x.size
    if n < 4:
        return (float('nan'), float('nan'), float('nan'))

    observed = float(stats.pearsonr(x, y)[0])

    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) == 0.0 or np.std(y[idx]) == 0.0:
            draws[i] = np.nan
            continue
        draws[i] = stats.pearsonr(x[idx], y[idx])[0]
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def association(feature: Sequence[float],
                performance: Sequence[float],
                label: str = 'pooled',
                n_boot: int = 5000,
                random_state: int | None = 0) -> Association:
    """Correlation between one connectivity feature and one performance measure."""
    x = np.asarray(feature, dtype=float)
    y = np.asarray(performance, dtype=float)
    if x.shape != y.shape:
        raise ValueError('feature and performance must have the same length')
    r, low, high = _bootstrap_r(x, y, n_boot, random_state)
    return Association(label, int(np.sum(np.isfinite(x) & np.isfinite(y))),
                       r, low, high)


def pooled_and_within(feature: Sequence[float],
                      performance: Sequence[float],
                      group: Sequence[str],
                      n_boot: int = 5000,
                      random_state: int | None = 0) -> pd.DataFrame:
    """Association pooled across groups and separately within each group.

    Always report both. A large pooled association alongside near-zero
    within-group associations means the correlation is carried by the group
    difference and says nothing about whether the feature tracks ability. The
    `interpretation` column states which case obtained.
    """
    x = np.asarray(feature, dtype=float)
    y = np.asarray(performance, dtype=float)
    g = np.asarray(group)
    if not (x.shape == y.shape == g.shape):
        raise ValueError('feature, performance and group must have the same length')

    rows = [association(x, y, 'pooled', n_boot, random_state)]
    for name in sorted(set(g.tolist())):
        mask = g == name
        rows.append(association(x[mask], y[mask], f'within:{name}',
                                n_boot, random_state))

    frame = pd.DataFrame([{
        'label': a.label, 'n': a.n, 'r': a.r,
        'ci_low': a.ci_low, 'ci_high': a.ci_high,
        'excludes_zero': a.excludes_zero,
    } for a in rows])

    pooled = rows[0]
    within = [a for a in rows[1:] if np.isfinite(a.r)]
    within_mean = float(np.mean([a.r for a in within])) if within else float('nan')

    # The verdict is driven by whether intervals exclude zero, not by a
    # magnitude threshold on r. At 16 participants per group the standard error
    # of a correlation is roughly 0.28, so any fixed cutoff on |r| is crossed by
    # chance routinely and would manufacture conclusions from noise.
    pooled_reliable = pooled.excludes_zero
    within_reliable = [a for a in within if a.excludes_zero]

    if not np.isfinite(pooled.r):
        verdict = 'undetermined: too few observations'
    elif within_reliable:
        names = ', '.join(a.label.split(':', 1)[1] for a in within_reliable)
        verdict = (f'feature tracks performance within group ({names}), '
                   'so the association is not merely a group difference')
    elif pooled_reliable:
        verdict = ('pooled association is carried by the group difference; '
                   'no within-group interval excludes zero, so there is no '
                   'evidence the feature tracks ability within group')
    else:
        verdict = ('no association detected pooled or within group; note that '
                   'this sample cannot exclude moderate associations, so this '
                   'is not evidence of absence')

    frame.attrs['interpretation'] = verdict
    frame.attrs['pooled_r'] = pooled.r
    frame.attrs['within_mean_r'] = within_mean
    frame.attrs['pooled_excludes_zero'] = pooled_reliable
    frame.attrs['n_within_excluding_zero'] = len(within_reliable)
    return frame


def feature_performance_table(features: Mapping[str, Sequence[float]],
                              performance: Mapping[str, Sequence[float]],
                              group: Sequence[str],
                              n_boot: int = 5000,
                              random_state: int | None = 0) -> pd.DataFrame:
    """Every connectivity feature against every performance measure.

    Returns a long-form table for the manuscript. Note this is a grid of tests,
    so intervals are descriptive; with 16 participants per group the design
    cannot support confirmatory inference here, and the results are reported as
    effect sizes rather than as p-values.
    """
    rows = []
    for f_name, f_values in features.items():
        for p_name, p_values in performance.items():
            frame = pooled_and_within(f_values, p_values, group,
                                      n_boot=n_boot, random_state=random_state)
            for _, record in frame.iterrows():
                rows.append({'feature': f_name, 'measure': p_name,
                             **record.to_dict()})
    out = pd.DataFrame(rows)
    out.attrs['note'] = ('descriptive effect sizes with bootstrap intervals; '
                         'no p-values, since n = 16 per group cannot support '
                         'confirmatory inference over a grid of comparisons')
    return out
