"""How much does each analytic choice matter?

Turns the vague claim that "methodological choices matter" into a quantitative
statement about which ones and by how much, by decomposing variance in accuracy
across the four specification dimensions.

Important caveat, and it belongs in the manuscript as well as here: every
specification is fitted to the same participants, so accuracies across
specifications are correlated and the decomposition mixes genuine sensitivity to
analytic choices with sampling noise common to all of them. Treat the components
as descriptive. They do not support significance tests, and no p-values are
returned here for that reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .specification import Specification

DIMENSIONS = ('atlas', 'statistic', 'aggregation', 'classifier', 'pretraining')


def _design_matrix(specs: Sequence[Specification],
                   dimensions: Sequence[str]) -> tuple[np.ndarray, dict[str, list[int]]]:
    """One-hot design with one level dropped per dimension for identifiability.

    Returns the matrix (with intercept) and a map from dimension name to the
    column indices it owns, so a dimension can be dropped for the partial fits.
    """
    columns: list[np.ndarray] = [np.ones(len(specs))]
    owned: dict[str, list[int]] = {}

    for dim in dimensions:
        levels = sorted({getattr(s, dim) for s in specs})
        if len(levels) < 2:
            owned[dim] = []
            continue
        owned[dim] = []
        for level in levels[1:]:  # drop first level as reference
            owned[dim].append(len(columns))
            columns.append(
                np.array([1.0 if getattr(s, dim) == level else 0.0 for s in specs]))

    return np.column_stack(columns), owned


def _r_squared(design: np.ndarray, y: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    total = float(np.sum((y - y.mean()) ** 2))

    # A constant y has no variance to explain and R2 is undefined, so return
    # zero. Testing `total == 0.0` is not enough: the mean of n identical floats
    # need not be exactly that float, and the error grows with n. At the 210
    # specifications of the balanced grid, the mean of 210 copies of 0.65 is one
    # ulp high, `total` lands at 2.6e-30 rather than 0, and the ratio below
    # divides a comparably tiny residual by it to produce a number in the
    # hundreds. The failure is silent and looks like an enormous effect.
    tolerance = (np.finfo(float).eps
                 * max(1.0, float(np.abs(y).max())) ** 2
                 * max(1, y.size))
    if total <= tolerance:
        return 0.0

    return float(1.0 - np.sum(residual ** 2) / total)


def variance_decomposition(accuracies: Mapping[str, float],
                           specifications: Sequence[Specification],
                           dimensions: Sequence[str] = DIMENSIONS) -> pd.DataFrame:
    """Partial and marginal variance in accuracy attributable to each dimension.

    Parameters
    ----------
    accuracies
        Mapping of spec_id to accuracy.
    specifications
        The Specification objects, used to recover each spec's levels.

    Returns
    -------
    DataFrame indexed by dimension with columns:
        marginal      R2 of that dimension alone
        partial       R2 lost when dropping it from the full model, i.e. the
                      variance it explains that no other dimension can
        n_levels      number of levels realised

    The `partial` column is the honest one for an unbalanced grid. The design is
    unbalanced by construction, because region-only atlases contribute
    region-level specifications only, so marginal values double-count shared
    variance and will not sum to the full model R2.
    """
    by_id = {s.spec_id: s for s in specifications}
    shared = [k for k in accuracies if k in by_id]
    if len(shared) < 4:
        raise ValueError('need at least four specifications to decompose')

    specs = [by_id[k] for k in shared]
    y = np.array([accuracies[k] for k in shared], dtype=float)

    # Detect the structural confound between atlas and aggregation. In the full
    # grid, region-only atlases appear at region scale only, so the two
    # dimensions are not orthogonal and variance cannot be cleanly apportioned
    # between them. Pass SpecificationRegistry().balanced to avoid this.
    realised: dict[str, set[str]] = {}
    for spec in specs:
        realised.setdefault(spec.atlas, set()).add(spec.aggregation)
    balanced = len({frozenset(v) for v in realised.values()}) <= 1

    design_full, owned = _design_matrix(specs, dimensions)
    r2_full = _r_squared(design_full, y)

    rows = []
    for dim in dimensions:
        levels = sorted({getattr(s, dim) for s in specs})

        if len(levels) < 2:
            rows.append({'dimension': dim, 'marginal': 0.0, 'partial': 0.0,
                         'n_levels': len(levels)})
            continue

        design_alone, _ = _design_matrix(specs, [dim])
        marginal = _r_squared(design_alone, y)

        drop = set(owned[dim])
        keep = [i for i in range(design_full.shape[1]) if i not in drop]
        r2_reduced = _r_squared(design_full[:, keep], y)

        rows.append({'dimension': dim,
                     'marginal': marginal,
                     'partial': max(0.0, r2_full - r2_reduced),
                     'n_levels': len(levels)})

    frame = pd.DataFrame(rows).set_index('dimension')
    frame.attrs['r2_full'] = r2_full
    frame.attrs['residual'] = 1.0 - r2_full
    frame.attrs['n_specifications'] = len(shared)
    frame.attrs['balanced'] = balanced
    frame.attrs['note'] = ('descriptive only: specifications share participants, '
                           'so components mix analytic sensitivity with common '
                           'sampling noise')
    if not balanced:
        frame.attrs['warning'] = (
            'unbalanced design: atlas and aggregation are structurally '
            'confounded because not every atlas realises every aggregation, so '
            'variance cannot be cleanly apportioned between them. Pass '
            'SpecificationRegistry().balanced for the orthogonal 210-'
            'specification subgrid.')
    return frame


def summarize(frame: pd.DataFrame) -> dict:
    """Flatten a decomposition into a dict suitable for the manuscript."""
    return {
        'r2_full': frame.attrs['r2_full'],
        'residual': frame.attrs['residual'],
        'n_specifications': frame.attrs['n_specifications'],
        'partial': frame['partial'].to_dict(),
        'marginal': frame['marginal'].to_dict(),
        'ranked_by_partial': frame['partial'].sort_values(ascending=False).index.tolist(),
        'balanced': frame.attrs['balanced'],
        'note': frame.attrs['note'],
        'warning': frame.attrs.get('warning'),
    }
