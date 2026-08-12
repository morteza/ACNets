"""Cross-specification agreement between model explanations.

This module carries the paper's central claim: that attribution is more stable
across the multiverse than accuracy is. If that holds, an explanation can be
trusted even where a point estimate of performance cannot.

Rank-biased overlap is the primary statistic rather than Kendall tau, because
what matters is whether the *top* of the ranking agrees. Two specifications
that both put CON-SMN and FPN-SMN first have agreed on the finding, regardless
of how they order the irrelevant tail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def rank_biased_overlap(left: Sequence[str],
                        right: Sequence[str],
                        p: float = 0.9) -> float:
    """Rank-biased overlap (Webber, Moffat and Zobel 2010), RBO_EXT.

    Top-weighted similarity between two ranked lists that need not contain the
    same items. p controls how much weight sits at the top: p = 0.9 puts about
    86% of the weight in the first 10 ranks.

    Returns 1.0 for identical orderings and 0.0 for disjoint lists.
    """
    if not 0.0 < p < 1.0:
        raise ValueError('p must be in (0, 1)')
    if len(left) == 0 and len(right) == 0:
        return 1.0
    if len(left) == 0 or len(right) == 0:
        return 0.0

    depth = min(len(left), len(right))
    seen_left: set[str] = set()
    seen_right: set[str] = set()
    overlap = 0
    total = 0.0

    for d in range(1, depth + 1):
        item_l, item_r = left[d - 1], right[d - 1]
        seen_left.add(item_l)
        seen_right.add(item_r)
        if item_l == item_r:
            overlap += 1
        else:
            if item_l in seen_right:
                overlap += 1
            if item_r in seen_left:
                overlap += 1
        total += (overlap / d) * (p ** d)

    # RBO_EXT: extrapolate the unseen tail assuming the observed rate continues
    return float((overlap / depth) * (p ** depth) + ((1.0 - p) / p) * total)


def kendall_agreement(left: Mapping[str, float],
                      right: Mapping[str, float]) -> float:
    """Kendall tau-b over features present in both attribution vectors.

    Unlike RBO this weights all ranks equally, so it is reported alongside RBO
    as a check that the conclusion is not an artefact of top-weighting.
    """
    shared = sorted(set(left) & set(right))
    if len(shared) < 3:
        return float('nan')
    a = [left[k] for k in shared]
    b = [right[k] for k in shared]
    tau, _ = stats.kendalltau(a, b)
    return float(tau)


def to_ranking(attribution: Mapping[str, float],
               top_k: int | None = None,
               absolute: bool = True) -> list[str]:
    """Feature names ordered by descending importance.

    absolute=True ranks by magnitude, which is the right choice for signed
    attributions such as Haufe patterns or SHAP values, where a strongly
    negative contribution is as important as a strongly positive one.
    """
    items = attribution.items()
    key = (lambda kv: abs(kv[1])) if absolute else (lambda kv: kv[1])
    ordered = [name for name, _ in sorted(items, key=key, reverse=True)]
    return ordered if top_k is None else ordered[:top_k]


@dataclass(frozen=True)
class AgreementResult:
    matrix: pd.DataFrame
    mean: float
    ci_low: float
    ci_high: float
    method: str
    n_specifications: int

    def summary(self) -> dict:
        return {
            'method': self.method,
            'mean_agreement': self.mean,
            'ci95': [self.ci_low, self.ci_high],
            'n_specifications': self.n_specifications,
        }


def agreement_matrix(attributions: Mapping[str, Mapping[str, float]],
                     method: str = 'rbo',
                     p: float = 0.9,
                     top_k: int | None = None,
                     n_boot: int = 5000,
                     random_state: int | None = 0) -> AgreementResult:
    """Pairwise agreement between the attribution vectors of many specifications.

    Parameters
    ----------
    attributions
        Mapping of spec_id to {feature_name: importance}.
    method
        'rbo' (top-weighted, primary) or 'kendall' (all ranks equal).
    top_k
        Truncate rankings before comparison. Useful when feature counts differ
        across specifications, e.g. 21 network edges against 12880 region edges.

    Returns
    -------
    AgreementResult
        Includes a bootstrap confidence interval on the mean off-diagonal
        agreement. The interval resamples specifications, not pairs, because
        pairs sharing a specification are not independent.
    """
    spec_ids = list(attributions)
    n = len(spec_ids)
    if n < 2:
        raise ValueError('need at least two specifications to compare')

    rankings = {s: to_ranking(attributions[s], top_k=top_k) for s in spec_ids}
    matrix = pd.DataFrame(np.eye(n), index=spec_ids, columns=spec_ids, dtype=float)

    for a, b in combinations(spec_ids, 2):
        if method == 'rbo':
            value = rank_biased_overlap(rankings[a], rankings[b], p=p)
        elif method == 'kendall':
            value = kendall_agreement(attributions[a], attributions[b])
        else:
            raise ValueError(f'unknown method: {method!r}')
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

    values = matrix.to_numpy()
    off_diagonal = values[~np.eye(n, dtype=bool)]
    mean = float(np.nanmean(off_diagonal))

    rng = np.random.default_rng(random_state)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = values[np.ix_(idx, idx)]
        # Exclude every pair that compares a specification with itself, not
        # merely the literal diagonal of the resampled matrix. Sampling n
        # specifications with replacement almost always draws at least one
        # twice -- at n = 12 it happens in essentially every draw -- and each
        # repeat places a self-comparison, which scores a perfect 1.0, at an
        # off-diagonal position. Masking only the diagonal lets those through
        # and biases the interval upward hard enough that it can sit entirely
        # above the point estimate it is supposed to bracket.
        same_specification = idx[:, None] == idx[None, :]
        kept = sub[~same_specification]
        boot[i] = np.nanmean(kept) if kept.size else np.nan
    low, high = np.nanquantile(boot, [0.025, 0.975])

    return AgreementResult(matrix, mean, float(low), float(high), method, n)


def random_agreement_baseline(feature_names: Sequence[str],
                              n_samples: int = 5000,
                              method: str = 'rbo',
                              p: float = 0.9,
                              top_k: int | None = None,
                              random_state: int | None = 0) -> np.ndarray:
    """Null distribution of agreement between random rankings of the same features.

    Raw agreement is NOT interpretable on its own, and this is the single
    easiest way to overstate the paper's thesis. Because every specification
    ranks the same feature vocabulary, two entirely random rankings already
    agree substantially under a top-weighted measure. Empirically, for RBO with
    p=0.9:

        6 features  -> random-pair RBO around 0.78
        21 features -> random-pair RBO around 0.43
        50 features -> random-pair RBO around 0.20

    So a reported RBO of 0.85 over 21 network edges is meaningful, while the
    same number over 6 features is close to chance. Always report agreement
    relative to this baseline, and always state the number of features.

    This is the attribution-space analogue of the argument in Combrisson and
    Jerbi (2015) about theoretical versus empirical chance in classification.
    """
    names = list(feature_names)
    rng = np.random.default_rng(random_state)
    out = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        left = list(rng.permutation(names))
        right = list(rng.permutation(names))
        if method == 'rbo':
            if top_k is not None:
                left, right = left[:top_k], right[:top_k]
            out[i] = rank_biased_overlap(left, right, p=p)
        elif method == 'kendall':
            scores_l = {n: float(len(names) - j) for j, n in enumerate(left)}
            scores_r = {n: float(len(names) - j) for j, n in enumerate(right)}
            out[i] = kendall_agreement(scores_l, scores_r)
        else:
            raise ValueError(f'unknown method: {method!r}')
    return out


def excess_agreement(observed: float, baseline: np.ndarray) -> dict:
    """Agreement in excess of what random rankings of the same features achieve.

    `normalized` rescales so that 0 means chance agreement and 1 means perfect
    agreement, which is the number to report in the abstract:

        normalized = (observed - baseline_mean) / (1 - baseline_mean)

    `p_value` is the upper-tail permutation p-value of the observed agreement
    against the random-ranking null.
    """
    baseline = np.asarray(baseline, dtype=float)
    mean = float(np.nanmean(baseline))
    denominator = 1.0 - mean
    normalized = ((observed - mean) / denominator
                  if abs(denominator) > 1e-12 else float('nan'))
    n_ge = int(np.sum(baseline >= observed))
    return {
        'observed': float(observed),
        'baseline_mean': mean,
        'baseline_p95': float(np.nanquantile(baseline, 0.95)),
        'normalized': float(normalized),
        'p_value': (1.0 + n_ge) / (1.0 + baseline.size),
    }


def selection_frequency(attributions: Mapping[str, Mapping[str, float]],
                        top_k: int = 5) -> pd.Series:
    """How often each feature appears in the top k across specifications.

    More honest than a single ranking at this sample size: it reports how
    reproducible a feature's prominence is, rather than asserting an order that
    32 subjects cannot support.
    """
    counts: dict[str, int] = {}
    for spec_id in attributions:
        for name in to_ranking(attributions[spec_id], top_k=top_k):
            counts[name] = counts.get(name, 0) + 1
    n = len(attributions)
    series = pd.Series(counts, dtype=float).sort_values(ascending=False) / n
    series.name = f'selection_frequency_top{top_k}'
    return series


def accuracy_agreement_baseline(n_subjects: int,
                                n_samples: int = 5000,
                                chance: float = 0.5,
                                random_state: int | None = 0) -> np.ndarray:
    """Null distribution of pairwise accuracy agreement between two chance models.

    The counterpart of random_agreement_baseline, needed so that accuracy and
    attribution stability can be compared on a common scale. Agreement between
    two accuracies is defined as 1 - |a_i - a_j|; under the null both are
    chance-level classifiers evaluated on n_subjects, so the expected absolute
    difference is set by binomial noise rather than by zero.
    """
    rng = np.random.default_rng(random_state)
    a = rng.binomial(n_subjects, chance, n_samples) / n_subjects
    b = rng.binomial(n_subjects, chance, n_samples) / n_subjects
    return 1.0 - np.abs(a - b)


def pairwise_accuracy_agreement(accuracies: Mapping[str, float]) -> float:
    """Mean pairwise accuracy agreement, 1 - |a_i - a_j|, over specifications."""
    values = np.asarray(list(accuracies.values()), dtype=float)
    if values.size < 2:
        raise ValueError('need at least two specifications')
    diffs = [abs(values[i] - values[j])
             for i, j in combinations(range(values.size), 2)]
    return float(1.0 - np.mean(diffs))


def agreement_vs_threshold(attributions: Mapping[str, Mapping[str, float]],
                           accuracies: Mapping[str, float],
                           thresholds: Sequence[float] | None = None,
                           method: str = 'rbo',
                           n_boot: int = 500,
                           **kwargs) -> pd.DataFrame:
    """Attribution agreement as a function of the accuracy cutoff used to select
    specifications.

    Conditioning on accuracy is not innocent: specifications that decode well
    may do so *because* they recovered the same signal, which inflates agreement
    among them. Reporting agreement at a single threshold hides that dependency.
    This returns the whole curve, including threshold 0 (all specifications), so
    a reader can see whether the conclusion is an artefact of the cutoff.
    """
    if thresholds is None:
        thresholds = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7]

    shared = sorted(set(attributions) & set(accuracies))
    rows = []
    for threshold in thresholds:
        keep = [s for s in shared if accuracies[s] >= threshold]
        if len(keep) < 2:
            rows.append({'threshold': threshold, 'n_specifications': len(keep),
                         'agreement': float('nan'), 'ci_low': float('nan'),
                         'ci_high': float('nan')})
            continue
        result = agreement_matrix({s: attributions[s] for s in keep},
                                  method=method, n_boot=n_boot, **kwargs)
        rows.append({'threshold': threshold, 'n_specifications': len(keep),
                     'agreement': result.mean, 'ci_low': result.ci_low,
                     'ci_high': result.ci_high})
    return pd.DataFrame(rows)


def stability_comparison(attributions: Mapping[str, Mapping[str, float]],
                         accuracies: Mapping[str, float],
                         n_subjects: int,
                         method: str = 'rbo',
                         n_boot: int = 2000,
                         random_state: int | None = 0,
                         **kwargs) -> dict:
    """Accuracy and attribution stability on a common excess-over-null scale.

    The paper's thesis is that explanation is more stable than prediction across
    the multiverse. Stating that requires the two to be measured comparably, and
    the obvious framing does not achieve it: comparing the coefficient of
    variation of a continuous accuracy against the agreement of a coarse ranking
    invites the objection that rankings only look stable because they are
    low-resolution.

    Both quantities are therefore expressed as pairwise agreement normalised
    against their own null, giving numbers in the same units where 0 is chance
    and 1 is perfect:

        attribution : mean pairwise RBO,        null = random rankings
        accuracy    : mean pairwise 1-|ai-aj|,  null = chance-level classifiers

    A positive `difference` supports the thesis. Because both nulls are
    empirical rather than analytic, neither number is inflated by the arbitrary
    floor of its own metric.
    """
    shared = sorted(set(attributions) & set(accuracies))
    if len(shared) < 2:
        raise ValueError('need at least two specifications with both measures')

    att = agreement_matrix({s: attributions[s] for s in shared},
                           method=method, n_boot=n_boot, **kwargs)
    vocabulary = sorted({f for s in shared for f in attributions[s]})
    att_null = random_agreement_baseline(
        vocabulary, method=method, p=kwargs.get('p', 0.9),
        top_k=kwargs.get('top_k'), random_state=random_state)
    att_excess = excess_agreement(att.mean, att_null)

    acc_observed = pairwise_accuracy_agreement({s: accuracies[s] for s in shared})
    acc_null = accuracy_agreement_baseline(n_subjects, random_state=random_state)
    acc_excess = excess_agreement(acc_observed, acc_null)

    return {
        'n_specifications': len(shared),
        'n_features': len(vocabulary),
        'n_subjects': n_subjects,
        'attribution': att_excess,
        'accuracy': acc_excess,
        'difference': att_excess['normalized'] - acc_excess['normalized'],
        'thesis_supported': bool(
            att_excess['normalized'] > acc_excess['normalized']),
        'method': method,
    }


def stability_vs_performance(attributions: Mapping[str, Mapping[str, float]],
                             accuracies: Mapping[str, float],
                             method: str = 'rbo',
                             **kwargs) -> dict:
    """Descriptive spread of accuracy alongside mean attribution agreement.

    Retained for reporting raw descriptives. For the paper's actual comparison
    use stability_comparison, which places both measures on a common
    excess-over-null scale; the coefficient of variation reported here is not
    commensurable with an agreement statistic.
    """
    shared = sorted(set(attributions) & set(accuracies))
    if len(shared) < 2:
        raise ValueError('need at least two specifications with both measures')

    result = agreement_matrix({s: attributions[s] for s in shared},
                              method=method, **kwargs)
    acc = np.array([accuracies[s] for s in shared], dtype=float)

    # Calibrate agreement against random rankings of the same feature
    # vocabulary. Without this the agreement number is uninterpretable, since
    # its floor depends on how many features there are.
    vocabulary = sorted({f for s in shared for f in attributions[s]})
    baseline = random_agreement_baseline(
        vocabulary,
        method=method,
        p=kwargs.get('p', 0.9),
        top_k=kwargs.get('top_k'),
        random_state=kwargs.get('random_state', 0),
    )
    calibrated = excess_agreement(result.mean, baseline)

    return {
        'n_specifications': len(shared),
        'n_features': len(vocabulary),
        'accuracy_mean': float(acc.mean()),
        'accuracy_sd': float(acc.std(ddof=1)),
        'accuracy_range': [float(acc.min()), float(acc.max())],
        'accuracy_cv': float(acc.std(ddof=1) / acc.mean()) if acc.mean() else float('nan'),
        'attribution_agreement': result.mean,
        'attribution_agreement_ci95': [result.ci_low, result.ci_high],
        'attribution_agreement_calibrated': calibrated,
        'method': method,
    }
