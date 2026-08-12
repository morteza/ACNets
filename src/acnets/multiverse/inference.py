"""Inference for multiverse classification results.

Three things here, each fixing a specific defect in the single-pipeline analysis:

1. accuracy_interval: uncertainty about accuracy given the number of subjects,
   not the spread across cross-validation splits. The latter shrinks as
   repetitions are added and is not a confidence interval on accuracy
   (Varoquaux 2018).
2. max_stat_permutation: a family-wise null over the whole specification space,
   so reporting the best of many paths does not inflate significance.
3. corrected_resampled_t: Nadeau and Bengio (2003), for comparing two
   specifications evaluated on overlapping resampled splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    method: str

    def __str__(self) -> str:
        return (f'{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] '
                f'({self.method})')


def accuracy_interval(n_correct: float,
                      n_total: int,
                      confidence: float = 0.95,
                      method: str = 'wilson') -> Interval:
    """Confidence interval for a classification accuracy.

    Parameters
    ----------
    n_correct
        Number of correct predictions. May be fractional when it is the mean
        accuracy times n_total, which is the usual case with repeated
        cross-validation.
    n_total
        Number of independent observations, i.e. the number of subjects. Not the
        number of folds times repetitions, which are not independent.
    method
        'wilson' (default) or 'clopper-pearson' (exact, more conservative).

    Notes
    -----
    For 32 subjects and 72.2% accuracy the Wilson interval is roughly
    [0.55, 0.85]. An interval computed across repeated CV splits on the same 32
    subjects can look like [0.70, 0.75], which understates the uncertainty by
    about an order of magnitude.
    """
    if n_total <= 0:
        raise ValueError('n_total must be positive')
    p_hat = n_correct / n_total
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError(f'accuracy out of range: {p_hat}')

    alpha = 1.0 - confidence

    if method == 'wilson':
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        z2 = z * z
        denom = 1.0 + z2 / n_total
        centre = (p_hat + z2 / (2.0 * n_total)) / denom
        margin = (z / denom) * np.sqrt(
            p_hat * (1.0 - p_hat) / n_total + z2 / (4.0 * n_total ** 2)
        )
        return Interval(p_hat, max(0.0, centre - margin),
                        min(1.0, centre + margin), 'Wilson')

    if method in ('clopper-pearson', 'exact'):
        k = n_correct
        low = 0.0 if k <= 0 else stats.beta.ppf(alpha / 2.0, k, n_total - k + 1)
        high = 1.0 if k >= n_total else stats.beta.ppf(
            1.0 - alpha / 2.0, k + 1, n_total - k)
        return Interval(p_hat, float(low), float(high), 'Clopper-Pearson')

    raise ValueError(f'unknown method: {method!r}')


@dataclass(frozen=True)
class AccuracyReport:
    """Two distinct uncertainties about one accuracy, kept separate.

    Repeated cross-validation and a small sample create different problems, and
    conflating them is the most common error in this literature. Both numbers
    below are legitimate; they answer different questions.

    `estimator_*` describes how much the estimate depends on the arbitrary
    choice of train/test split. Averaging over many repetitions is the correct
    remedy, and it works: the spread shrinks as repetitions are added, which is
    precisely why 100 runs are used here rather than one (Varoquaux et al. 2018;
    Chyzhyk et al. 2022). Reporting this quantity documents that the point
    estimate is not an artefact of one lucky partition.

    `sampling_*` describes how much the estimate depends on which participants
    happened to be recruited. Repetition cannot reduce it, because every
    repetition reuses the same people. It is governed by n and stays wide at
    n = 32 no matter how many splits are averaged.

    The failure mode is quoting `estimator_*` as though it were `sampling_*`.
    It is narrow, it looks like a confidence interval, and it shrinks with
    effort, so it is tempting. It says nothing about generalisation to new
    participants.
    """

    mean: float
    n_splits: int
    n_subjects: int
    estimator_sd: float
    estimator_low: float
    estimator_high: float
    sampling_low: float
    sampling_high: float

    @property
    def estimator_width(self) -> float:
        return self.estimator_high - self.estimator_low

    @property
    def sampling_width(self) -> float:
        return self.sampling_high - self.sampling_low

    def summary(self) -> dict:
        return {
            'mean': self.mean,
            'n_splits': self.n_splits,
            'n_subjects': self.n_subjects,
            'split_stability_sd': self.estimator_sd,
            'split_stability_interval': [self.estimator_low, self.estimator_high],
            'sampling_interval_wilson': [self.sampling_low, self.sampling_high],
            'width_ratio': (self.sampling_width / self.estimator_width
                            if self.estimator_width > 0 else float('inf')),
        }

    def __str__(self) -> str:
        return (f'accuracy {self.mean:.3f}; '
                f'split stability [{self.estimator_low:.3f}, {self.estimator_high:.3f}] '
                f'over {self.n_splits} splits; '
                f'sampling 95% CI [{self.sampling_low:.3f}, {self.sampling_high:.3f}] '
                f'at n = {self.n_subjects}')


def accuracy_report(fold_accuracies: np.ndarray,
                    n_subjects: int,
                    confidence: float = 0.95) -> AccuracyReport:
    """Summarise repeated-cross-validation accuracy without conflating variances.

    Parameters
    ----------
    fold_accuracies
        Accuracy from every fold of every run, e.g. 100 runs x 4 folds = 400
        values. Averaging these is the right way to get a stable point estimate.
    n_subjects
        Number of participants. Governs the sampling interval, and no amount of
        resampling changes it.

    Returns
    -------
    AccuracyReport
        Reports the mean, the across-split spread (labelled as split stability,
        not as a confidence interval), and the Wilson interval on n_subjects.

    Notes
    -----
    Report both. The across-split spread justifies the repeated-CV design and
    shows the estimate is not partition-dependent; the Wilson interval is what a
    reader needs in order to know how much the result would move in a new
    sample. Presenting only the first overstates precision; presenting only the
    second discards evidence that the design deliberately generated.
    """
    accuracies = np.asarray(fold_accuracies, dtype=float).ravel()
    if accuracies.size < 2:
        raise ValueError('need at least two fold accuracies')
    if np.any((accuracies < 0.0) | (accuracies > 1.0)):
        raise ValueError('accuracies must lie in [0, 1]')

    mean = float(accuracies.mean())
    sd = float(accuracies.std(ddof=1))
    alpha = 1.0 - confidence
    low, high = np.quantile(accuracies, [alpha / 2.0, 1.0 - alpha / 2.0])

    sampling = accuracy_interval(mean * n_subjects, n_subjects,
                                 confidence=confidence, method='wilson')

    return AccuracyReport(
        mean=mean,
        n_splits=int(accuracies.size),
        n_subjects=int(n_subjects),
        estimator_sd=sd,
        estimator_low=float(low),
        estimator_high=float(high),
        sampling_low=sampling.low,
        sampling_high=sampling.high,
    )


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p-value, upper tail.

    Uses the (1 + #{null >= observed}) / (1 + n) form so the p-value can never
    be zero and is correctly bounded below by 1 / (n_permutations + 1). A null
    of 100 permutations therefore cannot support p < 0.0099, which is worth
    checking before reporting a small p.
    """
    null = np.asarray(null, dtype=float)
    n_ge = int(np.sum(null >= observed))
    return (1.0 + n_ge) / (1.0 + null.size)


@dataclass(frozen=True)
class MaxStatResult:
    observed_max: float
    best_spec_id: str | None
    null_max: np.ndarray
    p_value: float
    n_permutations: int
    n_specifications: int

    @property
    def critical_value(self) -> float:
        """Accuracy a specification must exceed to be family-wise significant."""
        return float(np.quantile(self.null_max, 0.95))

    def summary(self) -> dict:
        return {
            'observed_max': self.observed_max,
            'best_spec_id': self.best_spec_id,
            'p_value': self.p_value,
            'critical_value_95': self.critical_value,
            'n_permutations': self.n_permutations,
            'n_specifications': self.n_specifications,
        }


def max_stat_permutation(observed: dict[str, float],
                         permuted: np.ndarray) -> MaxStatResult:
    """Family-wise null over the specification space.

    Parameters
    ----------
    observed
        Mapping of spec_id to observed accuracy, one entry per specification.
    permuted
        Array of shape (n_permutations, n_specifications) holding the accuracy
        of every specification under each label permutation. Column order must
        match the iteration order of `observed`.

    Returns
    -------
    MaxStatResult
        The p-value tests the largest observed accuracy against the
        distribution of the largest accuracy under the null. This is the
        correction for having searched the space, and it is exact rather than
        conservative like Bonferroni.

    Notes
    -----
    Labels must be permuted outside the entire pipeline, including feature
    selection and hyperparameter search. Permuting inside leaks the true
    labels through the selection step and produces an optimistic null.
    """
    permuted = np.asarray(permuted, dtype=float)
    if permuted.ndim != 2:
        raise ValueError('permuted must be 2d (n_permutations, n_specifications)')
    if permuted.shape[1] != len(observed):
        raise ValueError(
            f'permuted has {permuted.shape[1]} specifications but observed has '
            f'{len(observed)}')

    spec_ids = list(observed)
    values = np.array([observed[s] for s in spec_ids], dtype=float)
    best_index = int(np.argmax(values))
    observed_max = float(values[best_index])

    null_max = permuted.max(axis=1)
    return MaxStatResult(
        observed_max=observed_max,
        best_spec_id=spec_ids[best_index],
        null_max=null_max,
        p_value=permutation_pvalue(observed_max, null_max),
        n_permutations=permuted.shape[0],
        n_specifications=permuted.shape[1],
    )


def corrected_resampled_t(scores_a: np.ndarray,
                          scores_b: np.ndarray,
                          n_train: int,
                          n_test: int) -> tuple[float, float]:
    """Nadeau and Bengio (2003) corrected resampled t-test.

    Repeated random train/test splits reuse the same subjects, so per-split
    score differences are positively correlated and the naive paired t-test is
    anticonservative. The correction inflates the variance by
    (1/n + n_test/n_train).

    Returns (t_statistic, two_sided_p).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('score arrays must have the same shape')
    n = a.size
    if n < 2:
        raise ValueError('need at least two splits')

    diff = a - b
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)
    if var_diff == 0.0:
        return (0.0, 1.0)

    scale = (1.0 / n) + (n_test / n_train)
    t_stat = mean_diff / np.sqrt(scale * var_diff)
    p = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    return float(t_stat), float(p)


def specification_curve(observed: dict[str, float],
                        null_max: np.ndarray | None = None,
                        confidence: float = 0.95) -> list[dict]:
    """Sorted accuracies with family-wise significance flags.

    Returns records ordered by accuracy, ready for the specification curve
    figure. When null_max is supplied, `significant` marks specifications whose
    accuracy exceeds the family-wise critical value.
    """
    threshold = None
    if null_max is not None:
        threshold = float(np.quantile(np.asarray(null_max, dtype=float), confidence))

    records = [
        {'spec_id': spec_id, 'accuracy': acc,
         'significant': None if threshold is None else bool(acc > threshold)}
        for spec_id, acc in sorted(observed.items(), key=lambda kv: kv[1])
    ]
    for rank, record in enumerate(records, start=1):
        record['rank'] = rank
    return records
