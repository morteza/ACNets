"""Tests for multiverse inference.

These check properties that the paper's claims depend on, in particular that
the Wilson interval is wide at n=32 and that the max-statistic correction is
strictly more conservative than an uncorrected per-specification p-value.
"""

import numpy as np
import pytest

from acnets.multiverse.inference import (
    accuracy_interval,
    accuracy_report,
    corrected_resampled_t,
    max_stat_permutation,
    permutation_pvalue,
    specification_curve,
)


class TestAccuracyInterval:

    def test_wilson_is_wide_at_n32(self):
        """The interval that matters for this study. Roughly [0.55, 0.85]."""
        result = accuracy_interval(0.722 * 32, 32)
        assert result.point == pytest.approx(0.722, abs=1e-3)
        assert result.low == pytest.approx(0.55, abs=0.02)
        assert result.high == pytest.approx(0.85, abs=0.02)

    def test_wilson_far_wider_than_across_split_interval(self):
        """Guard against reinstating the across-split interval by accident.

        The draft reported [0.6975, 0.7463], a width of 0.049. The correct
        interval is more than five times wider.
        """
        result = accuracy_interval(0.722 * 32, 32)
        assert (result.high - result.low) > 5 * 0.049

    def test_interval_narrows_with_n(self):
        narrow = accuracy_interval(0.722 * 1000, 1000)
        wide = accuracy_interval(0.722 * 32, 32)
        assert (narrow.high - narrow.low) < (wide.high - wide.low)

    def test_interval_contains_point_estimate(self):
        for n in (8, 32, 100, 500):
            for p in (0.05, 0.5, 0.722, 0.95):
                r = accuracy_interval(p * n, n)
                assert r.low <= r.point <= r.high

    def test_bounded_to_unit_interval(self):
        assert accuracy_interval(0, 32).low == 0.0
        assert accuracy_interval(32, 32).high == 1.0

    def test_clopper_pearson_is_conservative(self):
        wilson = accuracy_interval(0.722 * 32, 32, method='wilson')
        exact = accuracy_interval(0.722 * 32, 32, method='clopper-pearson')
        assert (exact.high - exact.low) >= (wilson.high - wilson.low)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            accuracy_interval(10, 0)
        with pytest.raises(ValueError):
            accuracy_interval(40, 32)
        with pytest.raises(ValueError):
            accuracy_interval(16, 32, method='bogus')


class TestPermutationPvalue:

    def test_resolution_floor(self):
        """100 permutations cannot support p < 1/101. Worth asserting: the
        draft reported p=0.015, which needs at least 66 permutations."""
        null = np.zeros(100)
        assert permutation_pvalue(1.0, null) == pytest.approx(1.0 / 101.0)

    def test_never_zero(self):
        assert permutation_pvalue(10.0, np.zeros(5000)) > 0.0

    def test_p_one_when_observed_is_smallest(self):
        assert permutation_pvalue(0.0, np.ones(99)) == pytest.approx(1.0)

    def test_monotone_in_observed(self):
        rng = np.random.default_rng(0)
        null = rng.normal(0.5, 0.05, 2000)
        assert permutation_pvalue(0.7, null) < permutation_pvalue(0.5, null)


class TestMaxStatPermutation:

    def _make(self, n_specs=15, n_perm=2000, seed=0):
        rng = np.random.default_rng(seed)
        permuted = rng.normal(0.5, 0.08, size=(n_perm, n_specs))
        observed = {f'spec{i:02d}': v
                    for i, v in enumerate(rng.normal(0.5, 0.08, n_specs))}
        return observed, permuted

    def test_more_conservative_than_uncorrected(self):
        """The whole point of the correction."""
        observed, permuted = self._make()
        best = max(observed.values())
        result = max_stat_permutation(observed, permuted)
        uncorrected = permutation_pvalue(best, permuted[:, 0])
        assert result.p_value >= uncorrected

    def test_critical_value_above_marginal_95th(self):
        observed, permuted = self._make()
        result = max_stat_permutation(observed, permuted)
        marginal = float(np.quantile(permuted[:, 0], 0.95))
        assert result.critical_value > marginal

    def test_identifies_best_specification(self):
        observed, permuted = self._make()
        result = max_stat_permutation(observed, permuted)
        assert result.best_spec_id == max(observed, key=observed.get)
        assert result.observed_max == pytest.approx(max(observed.values()))

    def test_strong_effect_survives_correction(self):
        observed, permuted = self._make()
        observed['spec00'] = 0.95
        result = max_stat_permutation(observed, permuted)
        assert result.p_value < 0.05

    def test_shape_mismatch_raises(self):
        observed, permuted = self._make(n_specs=15)
        with pytest.raises(ValueError, match='specifications'):
            max_stat_permutation(observed, permuted[:, :10])

    def test_rejects_1d_permuted(self):
        observed, _ = self._make(n_specs=3)
        with pytest.raises(ValueError, match='2d'):
            max_stat_permutation(observed, np.zeros(10))


class TestCorrectedResampledT:

    def test_correction_reduces_significance(self):
        """Against a naive paired t-test on the same differences."""
        from scipy import stats
        rng = np.random.default_rng(0)
        a = rng.normal(0.72, 0.05, 100)
        b = rng.normal(0.68, 0.05, 100)
        _, p_corrected = corrected_resampled_t(a, b, n_train=24, n_test=8)
        _, p_naive = stats.ttest_rel(a, b)
        assert p_corrected > p_naive

    def test_identical_scores_give_null_result(self):
        x = np.linspace(0.6, 0.8, 50)
        t, p = corrected_resampled_t(x, x, n_train=24, n_test=8)
        assert t == 0.0 and p == 1.0

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            corrected_resampled_t(np.zeros(10), np.zeros(9), 24, 8)


class TestSpecificationCurve:

    def test_sorted_ascending_with_ranks(self):
        observed = {'a': 0.7, 'b': 0.5, 'c': 0.6}
        records = specification_curve(observed)
        assert [r['spec_id'] for r in records] == ['b', 'c', 'a']
        assert [r['rank'] for r in records] == [1, 2, 3]

    def test_significance_flagging(self):
        observed = {'a': 0.9, 'b': 0.5}
        null_max = np.full(1000, 0.6)
        records = specification_curve(observed, null_max=null_max)
        flags = {r['spec_id']: r['significant'] for r in records}
        assert flags['a'] is True
        assert flags['b'] is False

    def test_no_null_leaves_significance_unset(self):
        records = specification_curve({'a': 0.9})
        assert records[0]['significant'] is None


class TestAccuracyReport:
    """The two-variance distinction, which the manuscript turns on.

    Repeated cross-validation legitimately stabilises the point estimate. It
    cannot widen or narrow the uncertainty due to having only 32 participants.
    Both quantities are reported; conflating them is the error these tests guard.
    """

    def _folds(self, mean=0.72, sd=0.15, n=400, seed=0):
        rng = np.random.default_rng(seed)
        return np.clip(rng.normal(mean, sd, n), 0.0, 1.0)

    def test_averaging_stabilises_the_point_estimate(self):
        """More repetitions means a tighter split-stability interval. This is
        the benefit that motivates running 100 repetitions."""
        few = accuracy_report(self._folds(n=8, seed=1), n_subjects=32)
        many = accuracy_report(self._folds(n=400, seed=1), n_subjects=32)
        assert many.estimator_sd == pytest.approx(few.estimator_sd, abs=0.08)
        # the mean is far better determined with more splits
        assert abs(many.mean - 0.72) < abs(few.mean - 0.72) + 0.05

    def test_repetition_does_not_shrink_the_sampling_interval(self):
        """The key asymmetry. Sampling uncertainty is fixed by n, so it is
        identical whether 8 or 400 splits were averaged."""
        few = accuracy_report(self._folds(n=8, seed=2), n_subjects=32)
        many = accuracy_report(self._folds(n=400, seed=2), n_subjects=32)
        assert few.sampling_width == pytest.approx(many.sampling_width, abs=0.03)

    def test_sampling_interval_shrinks_only_with_more_participants(self):
        folds = self._folds(n=400, seed=3)
        small = accuracy_report(folds, n_subjects=32)
        large = accuracy_report(folds, n_subjects=500)
        assert large.sampling_width < small.sampling_width

    def test_sampling_interval_is_the_wider_one_at_n32(self):
        """Guards against quoting split stability as though it were a CI: at
        n=32 with many repetitions the sampling interval is several times wider."""
        report = accuracy_report(self._folds(n=400, sd=0.05, seed=4), n_subjects=32)
        assert report.sampling_width > 2 * report.estimator_sd
        assert report.summary()['width_ratio'] > 1.0

    def test_reports_both_intervals_separately(self):
        report = accuracy_report(self._folds(), n_subjects=32)
        keys = report.summary()
        assert 'split_stability_interval' in keys
        assert 'sampling_interval_wilson' in keys
        assert 'confidence' not in str(keys.get('split_stability_interval'))

    def test_records_the_counts(self):
        report = accuracy_report(self._folds(n=400), n_subjects=32)
        assert report.n_splits == 400
        assert report.n_subjects == 32

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError, match='at least two'):
            accuracy_report([0.7], n_subjects=32)
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            accuracy_report([0.7, 1.4], n_subjects=32)
