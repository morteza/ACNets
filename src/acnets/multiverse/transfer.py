"""Fold-independent transforms fitted on an external resting dataset.

This is the classic-ML half of the transfer idea, and the reason it earns a
place in the multiverse rather than in a footnote: it isolates how much of any
deep model's gain is *the external data* rather than the depth. A
`pretraining='lemon'` specification differs from its `'none'` twin in exactly
two places, both fitted on MPI-LEMON and frozen before Julia2018 is touched:

1. the connectivity measure. For `tangent` this is substantive — the tangent
   statistic projects each subject's covariance onto a reference point, and
   estimating that reference from 220 external subjects instead of 24 training
   subjects is both a better estimate and a leakage-proof one. For the other
   statistics `fit` is stateless, so the measure is shared for uniformity and
   the twin specifications differ only in scaling.
2. the feature scaler, fitted on LEMON's feature distribution instead of the
   training fold's.

Leakage argument, stated once: LEMON and Julia2018 share no subjects, and
nothing here ever receives a label — `fit_lemon_transforms` takes timeseries
only. A specification using these transforms therefore cannot leak Julia2018's
group structure through them, which a test asserts by construction
(`transfer_test.py`).

The 652-timepoint LEMON runs are cut into windows matching Julia2018's
124-timepoint length. That serves two purposes: it multiplies the sample
(220 subjects → ~1,100 windows) and it removes the run-length mismatch, since
several statistics are sensitive to the number of samples behind each matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..connectome import ExtraConnectivityMeasure
from .runner import vectorize
from .specification import ASYMMETRIC_STATISTICS


def windowed(timeseries: np.ndarray,
             length: int,
             stride: int | None = None) -> np.ndarray:
    """Cut (n_subjects, n_timepoints, n_nodes) into windows of `length`.

    Returns (n_subjects * k, length, n_nodes). Windows are per-subject
    z-scored, which is also the TR/site-mismatch mitigation: after
    standardisation the statistics see fluctuation structure, not units.
    """
    timeseries = np.asarray(timeseries, dtype=float)
    if timeseries.ndim != 3:
        raise ValueError(f'expected 3d (subjects, timepoints, nodes), '
                         f'got {timeseries.shape}')
    n_timepoints = timeseries.shape[1]
    if length > n_timepoints:
        raise ValueError(f'window length {length} exceeds run length {n_timepoints}')
    stride = stride or length  # non-overlapping by default

    starts = range(0, n_timepoints - length + 1, stride)
    windows = np.concatenate(
        [timeseries[:, start:start + length, :] for start in starts], axis=0)

    mean = windows.mean(axis=1, keepdims=True)
    sd = windows.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (windows - mean) / sd


@dataclass(frozen=True)
class LemonTransforms:
    """Connectivity measures and scalers fitted on LEMON, keyed by statistic."""

    measures: dict[str, ExtraConnectivityMeasure]
    scalers: dict[str, StandardScaler]
    node_names: tuple[str, ...]
    n_windows: int

    def measure(self, statistic: str) -> ExtraConnectivityMeasure:
        if statistic not in self.measures:
            raise KeyError(f'no LEMON transform fitted for {statistic!r}; '
                           f'have {sorted(self.measures)}')
        return self.measures[statistic]

    def scaler(self, statistic: str) -> StandardScaler:
        return self.scalers[statistic]

    def check_nodes(self, node_names: Sequence[str]) -> None:
        """The transforms only mean anything over the same feature vocabulary."""
        if tuple(node_names) != self.node_names:
            raise ValueError(
                'node vocabulary mismatch between LEMON transforms and the '
                f'target data: {self.node_names} vs {tuple(node_names)}')


def fit_lemon_transforms(timeseries: np.ndarray,
                         node_names: Sequence[str],
                         statistics: Sequence[str],
                         window_length: int = 124,
                         stride: int | None = None) -> LemonTransforms:
    """Fit per-statistic measures and scalers on LEMON windows.

    Parameters
    ----------
    timeseries
        LEMON timeseries, (n_subjects, n_timepoints, n_nodes), already
        aggregated to the scale the target specifications use.
    node_names
        Names of the nodes, recorded so the target run can assert vocabulary
        identity rather than trusting it.
    statistics
        Which connectivity statistics to fit. Labels are never an input.
    """
    windows = windowed(timeseries, window_length, stride)

    measures: dict[str, ExtraConnectivityMeasure] = {}
    scalers: dict[str, StandardScaler] = {}
    for statistic in statistics:
        measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
        measure.fit(windows)
        features = vectorize(measure.transform(windows),
                             statistic in ASYMMETRIC_STATISTICS)
        measures[statistic] = measure
        scalers[statistic] = StandardScaler().fit(features)

    return LemonTransforms(measures=measures, scalers=scalers,
                           node_names=tuple(str(n) for n in node_names),
                           n_windows=int(windows.shape[0]))
