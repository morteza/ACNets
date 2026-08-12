"""Tests for cross-specification attribution agreement.

The RBO tests check the closed-form properties of Webber et al. (2010) RBO_EXT.
The stability tests check the paper's thesis is detectable when true and absent
when false, which guards against a metric that reports agreement regardless.
"""

import numpy as np
import pytest

from acnets.multiverse.agreement import (
    accuracy_agreement_baseline,
    agreement_matrix,
    agreement_vs_threshold,
    excess_agreement,
    kendall_agreement,
    pairwise_accuracy_agreement,
    random_agreement_baseline,
    rank_biased_overlap,
    selection_frequency,
    stability_comparison,
    stability_vs_performance,
    to_ranking,
)


class TestRankBiasedOverlap:

    def test_identical_lists_give_one(self):
        items = ['CON-SMN', 'FPN-SMN', 'CON-FPN', 'DMN-SMN']
        assert rank_biased_overlap(items, items) == pytest.approx(1.0)

    def test_disjoint_lists_give_zero(self):
        a = rank_biased_overlap(['a', 'b', 'c'], ['x', 'y', 'z'])
        assert a == pytest.approx(0.0)

    def test_reversed_lists_score_below_identical(self):
        """Reversed lists score lower than identical ones, but NOT near zero.

        RBO_EXT for two reversed permutations of the same 8 items is 0.5895
        (verified by hand against Webber et al. 2010 eq. 32). The value is high
        because RBO_EXT extrapolates the tail, and two permutations of a shared
        item set reach complete overlap at full depth. This is why raw RBO must
        always be calibrated against random_agreement_baseline.
        """
        items = list('abcdefgh')
        reversed_score = rank_biased_overlap(items, items[::-1])
        assert reversed_score == pytest.approx(0.589459, abs=1e-5)
        assert reversed_score < rank_biased_overlap(items, items)

    def test_symmetric(self):
        a, b = list('abcdef'), list('badcfe')
        assert rank_biased_overlap(a, b) == pytest.approx(
            rank_biased_overlap(b, a))

    def test_top_weighted(self):
        """Agreeing at the top beats agreeing at the bottom.

        This is why RBO is the primary statistic: two specifications that both
        rank CON-SMN first have agreed on the finding.
        """
        reference = list('abcdefgh')
        top_agree = ['a', 'b', 'c', 'd', 'h', 'g', 'f', 'e']
        bottom_agree = ['d', 'c', 'b', 'a', 'e', 'f', 'g', 'h']
        assert (rank_biased_overlap(reference, top_agree)
                > rank_biased_overlap(reference, bottom_agree))

    def test_p_controls_top_weighting(self):
        reference = list('abcdefghij')
        swapped_tail = list('abcdefghji')
        low_p = rank_biased_overlap(reference, swapped_tail, p=0.5)
        high_p = rank_biased_overlap(reference, swapped_tail, p=0.99)
        assert low_p > high_p

    def test_bounded(self):
        rng = np.random.default_rng(0)
        items = [f'f{i}' for i in range(20)]
        for _ in range(50):
            a = list(rng.permutation(items))
            b = list(rng.permutation(items))
            assert 0.0 <= rank_biased_overlap(a, b) <= 1.0

    def test_empty_lists(self):
        assert rank_biased_overlap([], []) == pytest.approx(1.0)
        assert rank_biased_overlap(['a'], []) == pytest.approx(0.0)

    def test_rejects_bad_p(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                rank_biased_overlap(['a'], ['a'], p=bad)


class TestToRanking:

    def test_orders_by_magnitude_by_default(self):
        """Signed attributions must rank by absolute value: a strongly negative
        Haufe pattern is as informative as a strongly positive one."""
        attribution = {'a': 0.1, 'b': -0.9, 'c': 0.5}
        assert to_ranking(attribution) == ['b', 'c', 'a']

    def test_signed_ordering_when_requested(self):
        attribution = {'a': 0.1, 'b': -0.9, 'c': 0.5}
        assert to_ranking(attribution, absolute=False) == ['c', 'a', 'b']

    def test_top_k_truncates(self):
        attribution = {c: float(i) for i, c in enumerate('abcdef')}
        assert len(to_ranking(attribution, top_k=3)) == 3


class TestKendallAgreement:

    def test_identical_gives_one(self):
        att = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
        assert kendall_agreement(att, att) == pytest.approx(1.0)

    def test_reversed_gives_minus_one(self):
        a = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
        b = {'a': 4.0, 'b': 3.0, 'c': 2.0, 'd': 1.0}
        assert kendall_agreement(a, b) == pytest.approx(-1.0)

    def test_too_few_shared_features_gives_nan(self):
        assert np.isnan(kendall_agreement({'a': 1.0}, {'a': 1.0}))


class TestAgreementMatrix:

    def _consistent(self, n=8, seed=0):
        """Specifications that agree on the top two features."""
        rng = np.random.default_rng(seed)
        features = ['CON-SMN', 'FPN-SMN', 'CON-FPN', 'DMN-SMN', 'DMN-FPN', 'OCC-SMN']
        out = {}
        for i in range(n):
            base = np.array([1.0, 0.9, 0.3, 0.2, 0.1, 0.05])
            out[f'spec{i:02d}'] = dict(zip(features,
                                           base + rng.normal(0, 0.02, base.size), strict=False))
        return out

    def _inconsistent(self, n=8, seed=0):
        rng = np.random.default_rng(seed)
        features = ['CON-SMN', 'FPN-SMN', 'CON-FPN', 'DMN-SMN', 'DMN-FPN', 'OCC-SMN']
        return {f'spec{i:02d}': dict(zip(features, rng.normal(0, 1, len(features)), strict=False))
                for i in range(n)}

    def test_matrix_is_symmetric_with_unit_diagonal(self):
        result = agreement_matrix(self._consistent(), n_boot=200)
        values = result.matrix.to_numpy()
        assert np.allclose(values, values.T)
        assert np.allclose(np.diag(values), 1.0)

    def test_consistent_scores_above_inconsistent(self):
        high = agreement_matrix(self._consistent(), n_boot=500).mean
        low = agreement_matrix(self._inconsistent(), n_boot=500).mean
        assert high > low

    def test_bootstrap_interval_brackets_mean(self):
        result = agreement_matrix(self._consistent(), n_boot=500)
        assert result.ci_low <= result.mean <= result.ci_high

    def test_bootstrap_interval_brackets_mean_when_agreement_is_low(self):
        """The consistent fixture agrees so strongly it hides bootstrap bias.

        Every resample of n specifications drawn with replacement repeats at
        least one of them, and a repeat compares a specification with itself.
        Letting those perfect scores into the off-diagonal mean pushes the
        interval up; when true agreement is already near 1.0 there is no room
        for it to show, so the bias only surfaces on a set that disagrees.
        """
        result = agreement_matrix(self._inconsistent(n=12), n_boot=2000)
        assert result.ci_low <= result.mean <= result.ci_high

    def test_bootstrap_ignores_self_comparisons(self):
        """A resample is a mean over distinct pairs, so it cannot exceed the max."""
        attributions = self._inconsistent(n=12)
        result = agreement_matrix(attributions, n_boot=2000)
        values = result.matrix.to_numpy()
        off_diagonal = values[~np.eye(values.shape[0], dtype=bool)]
        assert result.ci_high <= off_diagonal.max()

    def test_kendall_method_runs(self):
        result = agreement_matrix(self._consistent(), method='kendall', n_boot=200)
        assert -1.0 <= result.mean <= 1.0

    def test_requires_two_specifications(self):
        with pytest.raises(ValueError, match='at least two'):
            agreement_matrix({'only': {'a': 1.0}})

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match='unknown method'):
            agreement_matrix(self._consistent(), method='bogus', n_boot=10)


class TestSelectionFrequency:

    def test_invariant_feature_reaches_one(self):
        attributions = {
            f'spec{i}': {'CON-SMN': 1.0, 'FPN-SMN': 0.9, 'other': 0.1}
            for i in range(10)
        }
        freq = selection_frequency(attributions, top_k=2)
        assert freq['CON-SMN'] == pytest.approx(1.0)
        assert 'other' not in freq

    def test_frequencies_bounded(self):
        rng = np.random.default_rng(0)
        features = [f'f{i}' for i in range(10)]
        attributions = {f'spec{i}': dict(zip(features, rng.normal(0, 1, 10), strict=False))
                        for i in range(20)}
        freq = selection_frequency(attributions, top_k=3)
        assert freq.between(0.0, 1.0).all()


class TestRandomAgreementBaseline:
    """The calibration that keeps the paper's headline claim honest."""

    def test_baseline_depends_strongly_on_feature_count(self):
        """Fewer features means a much higher chance floor.

        A reported RBO of 0.8 is strong evidence over 50 features and close to
        worthless over 6. Any agreement number must be accompanied by n_features.
        """
        small = random_agreement_baseline([f'f{i}' for i in range(6)], n_samples=1500)
        medium = random_agreement_baseline([f'f{i}' for i in range(21)], n_samples=1500)
        large = random_agreement_baseline([f'f{i}' for i in range(50)], n_samples=1500)
        assert small.mean() > medium.mean() > large.mean()
        assert small.mean() > 0.7
        assert large.mean() < 0.3

    def test_network_level_floor_is_substantial(self):
        """21 edges is the paper's network-level case: chance RBO is about 0.43,
        so raw agreement below roughly 0.6 carries no evidential weight."""
        baseline = random_agreement_baseline(
            [f'edge{i}' for i in range(21)], n_samples=3000)
        assert baseline.mean() == pytest.approx(0.43, abs=0.05)

    def test_kendall_baseline_centred_on_zero(self):
        baseline = random_agreement_baseline(
            [f'f{i}' for i in range(20)], n_samples=1000, method='kendall')
        assert baseline.mean() == pytest.approx(0.0, abs=0.05)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match='unknown method'):
            random_agreement_baseline(['a', 'b'], n_samples=5, method='bogus')


class TestExcessAgreement:

    def test_chance_agreement_normalizes_to_zero(self):
        baseline = np.full(1000, 0.43)
        assert excess_agreement(0.43, baseline)['normalized'] == pytest.approx(0.0)

    def test_perfect_agreement_normalizes_to_one(self):
        baseline = np.full(1000, 0.43)
        assert excess_agreement(1.0, baseline)['normalized'] == pytest.approx(1.0)

    def test_high_raw_agreement_can_be_unimpressive(self):
        """0.80 raw looks strong but is near chance when the floor is 0.78."""
        six_feature_floor = np.full(2000, 0.78)
        result = excess_agreement(0.80, six_feature_floor)
        assert result['normalized'] < 0.15

    def test_same_raw_value_is_strong_with_a_low_floor(self):
        fifty_feature_floor = np.full(2000, 0.20)
        result = excess_agreement(0.80, fifty_feature_floor)
        assert result['normalized'] > 0.7

    def test_pvalue_bounded_below(self):
        baseline = np.zeros(999)
        assert excess_agreement(1.0, baseline)['p_value'] == pytest.approx(0.001)


class TestStabilityVsPerformance:

    def test_detects_the_papers_thesis(self):
        """Accuracy varies a lot, attribution barely: the claimed pattern."""
        rng = np.random.default_rng(0)
        features = ['CON-SMN', 'FPN-SMN', 'CON-FPN', 'DMN-SMN']
        base = np.array([1.0, 0.9, 0.3, 0.2])
        attributions, accuracies = {}, {}
        for i in range(12):
            key = f'spec{i:02d}'
            attributions[key] = dict(zip(features, base + rng.normal(0, 0.02, 4), strict=False))
            accuracies[key] = float(rng.uniform(0.5, 0.75))

        out = stability_vs_performance(attributions, accuracies, n_boot=300)
        assert out['accuracy_cv'] > 0.05
        assert out['attribution_agreement'] > 0.9
        assert out['n_specifications'] == 12

    def test_requires_overlapping_keys(self):
        with pytest.raises(ValueError, match='at least two'):
            stability_vs_performance({'a': {'f': 1.0}}, {'b': 0.5}, n_boot=10)


class TestStabilityComparison:
    """The apples-to-apples version of the paper's central claim.

    Comparing the coefficient of variation of a continuous accuracy against the
    agreement of a coarse ranking is not a fair test: a reviewer can object that
    rankings only look stable because they are low-resolution. These tests check
    that placing both on a common excess-over-null scale detects the thesis when
    true and rejects it when false.
    """

    def _features(self):
        return ['CON-SMN', 'FPN-SMN', 'CON-FPN', 'DMN-SMN', 'DMN-FPN', 'OCC-SMN']

    def _thesis_true(self, n=14, seed=0):
        """Attribution nearly identical, accuracy all over the place."""
        rng = np.random.default_rng(seed)
        base = np.array([1.0, 0.9, 0.3, 0.2, 0.1, 0.05])
        att, acc = {}, {}
        for i in range(n):
            key = f'spec{i:02d}'
            att[key] = dict(zip(self._features(), base + rng.normal(0, 0.01, 6), strict=False))
            acc[key] = float(rng.uniform(0.50, 0.78))
        return att, acc

    def _thesis_false(self, n=14, seed=0):
        """Accuracy tightly clustered, attribution rankings unrelated."""
        rng = np.random.default_rng(seed)
        att, acc = {}, {}
        for i in range(n):
            key = f'spec{i:02d}'
            att[key] = dict(zip(self._features(), rng.normal(0, 1, 6), strict=False))
            acc[key] = float(0.72 + rng.normal(0, 0.002))
        return att, acc

    def test_detects_thesis_when_true(self):
        att, acc = self._thesis_true()
        out = stability_comparison(att, acc, n_subjects=32, n_boot=300)
        assert out['thesis_supported'] is True
        assert out['difference'] > 0

    def test_rejects_thesis_when_false(self):
        att, acc = self._thesis_false()
        out = stability_comparison(att, acc, n_subjects=32, n_boot=300)
        assert out['thesis_supported'] is False
        assert out['difference'] < 0

    def test_both_measures_on_common_scale(self):
        att, acc = self._thesis_true()
        out = stability_comparison(att, acc, n_subjects=32, n_boot=300)
        for side in ('attribution', 'accuracy'):
            assert 'normalized' in out[side]
            assert out[side]['normalized'] <= 1.0

    def test_reports_feature_and_subject_counts(self):
        att, acc = self._thesis_true()
        out = stability_comparison(att, acc, n_subjects=32, n_boot=200)
        assert out['n_features'] == 6
        assert out['n_subjects'] == 32
        assert out['n_specifications'] == 14


class TestAccuracyAgreementBaseline:

    def test_null_agreement_below_one(self):
        """Two chance classifiers on 32 subjects do not agree perfectly."""
        baseline = accuracy_agreement_baseline(32, n_samples=4000)
        assert baseline.mean() < 1.0
        assert baseline.mean() > 0.8

    def test_larger_n_gives_tighter_null(self):
        small = accuracy_agreement_baseline(16, n_samples=4000)
        large = accuracy_agreement_baseline(1000, n_samples=4000)
        assert large.mean() > small.mean()

    def test_pairwise_accuracy_agreement_bounds(self):
        assert pairwise_accuracy_agreement({'a': 0.7, 'b': 0.7}) == pytest.approx(1.0)
        assert pairwise_accuracy_agreement({'a': 0.0, 'b': 1.0}) == pytest.approx(0.0)

    def test_pairwise_requires_two(self):
        with pytest.raises(ValueError, match='at least two'):
            pairwise_accuracy_agreement({'a': 0.7})


class TestAgreementVsThreshold:
    """Conditioning on accuracy inflates agreement; the curve exposes it."""

    def test_returns_row_per_threshold(self):
        rng = np.random.default_rng(0)
        features = [f'f{i}' for i in range(8)]
        att = {f'spec{i:02d}': dict(zip(features, rng.normal(0, 1, 8), strict=False))
               for i in range(20)}
        acc = {k: float(rng.uniform(0.45, 0.8)) for k in att}
        curve = agreement_vs_threshold(att, acc, thresholds=[0.0, 0.6, 0.7],
                                       n_boot=100)
        assert list(curve['threshold']) == [0.0, 0.6, 0.7]
        assert curve['n_specifications'].is_monotonic_decreasing

    def test_includes_all_specifications_at_threshold_zero(self):
        rng = np.random.default_rng(1)
        features = [f'f{i}' for i in range(6)]
        att = {f'spec{i}': dict(zip(features, rng.normal(0, 1, 6), strict=True))
               for i in range(10)}
        acc = {k: 0.6 for k in att}
        curve = agreement_vs_threshold(att, acc, thresholds=[0.0], n_boot=100)
        assert curve.loc[0, 'n_specifications'] == 10

    def test_nan_when_too_few_survive(self):
        att = {'a': {'f1': 1.0, 'f2': 2.0}, 'b': {'f1': 2.0, 'f2': 1.0}}
        acc = {'a': 0.5, 'b': 0.5}
        curve = agreement_vs_threshold(att, acc, thresholds=[0.9], n_boot=50)
        assert np.isnan(curve.loc[0, 'agreement'])
