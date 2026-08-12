"""Tests for the brain-behaviour association analysis.

The important tests are the Simpson's paradox ones: a two-group design can
manufacture a large pooled correlation from a pure group difference, and the
module must say so rather than report the pooled value as a brain-behaviour
finding.
"""

import numpy as np
import pytest

from acnets.analysis.brain_behavior import (
    association,
    feature_performance_table,
    pooled_and_within,
)


class TestAssociation:

    def test_recovers_a_known_correlation(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        y = 0.8 * x + rng.normal(0, 0.6, 200)
        result = association(x, y, n_boot=800)
        assert 0.5 < result.r < 0.95
        assert result.ci_low <= result.r <= result.ci_high
        assert result.excludes_zero

    def test_no_association_interval_spans_zero(self):
        rng = np.random.default_rng(1)
        result = association(rng.normal(0, 1, 200), rng.normal(0, 1, 200), n_boot=800)
        assert not result.excludes_zero

    def test_too_few_observations_gives_nan(self):
        result = association([1.0, 2.0], [1.0, 2.0], n_boot=100)
        assert np.isnan(result.r)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match='same length'):
            association([1.0, 2.0, 3.0], [1.0, 2.0], n_boot=10)


class TestPooledAndWithin:
    """Guards the inference the manuscript depends on."""

    def test_flags_group_driven_pooled_correlation(self):
        """The trap. AVGPs have higher coupling AND better performance, but
        within each group the two are unrelated. The pooled r is large and
        entirely artefactual; the module must say so."""
        rng = np.random.default_rng(0)
        n = 16
        coupling = np.concatenate([rng.normal(0.6, 0.05, n),   # AVGP
                                   rng.normal(0.3, 0.05, n)])  # NVGP
        perf = np.concatenate([rng.normal(0.85, 0.03, n),
                               rng.normal(0.70, 0.03, n)])
        group = np.array(['AVGP'] * n + ['NVGP'] * n)

        frame = pooled_and_within(coupling, perf, group, n_boot=600)
        assert abs(frame.attrs['pooled_r']) > 0.6
        assert abs(frame.attrs['within_mean_r']) < 0.3
        assert 'carried by the group difference' in frame.attrs['interpretation']

    def test_detects_a_genuine_within_group_association(self):
        rng = np.random.default_rng(2)
        n = 40
        coupling = np.concatenate([rng.normal(0.6, 0.15, n),
                                   rng.normal(0.3, 0.15, n)])
        noise = rng.normal(0, 0.02, 2 * n)
        perf = 0.5 * coupling + noise           # real within-group relation
        group = np.array(['AVGP'] * n + ['NVGP'] * n)

        frame = pooled_and_within(coupling, perf, group, n_boot=600)
        assert abs(frame.attrs['within_mean_r']) > 0.5
        assert 'within group' in frame.attrs['interpretation']

    def test_reports_no_association_when_there_is_none(self):
        """And says so honestly: a null at this n is not evidence of absence."""
        rng = np.random.default_rng(3)
        n = 30
        frame = pooled_and_within(rng.normal(0, 1, 2 * n), rng.normal(0, 1, 2 * n),
                                  np.array(['AVGP'] * n + ['NVGP'] * n), n_boot=600)
        assert 'no association detected' in frame.attrs['interpretation']
        assert 'not evidence of absence' in frame.attrs['interpretation']

    def test_verdict_uses_intervals_not_magnitude_thresholds(self):
        """At n=16 per group the standard error of r is about 0.28, so a fixed
        cutoff on |r| would fire on noise. Random data must not yield a
        within-group verdict however large the sample r happens to be."""
        np.random.default_rng(7)
        n = 16
        group = np.array(['AVGP'] * n + ['NVGP'] * n)
        for seed in range(6):
            r = np.random.default_rng(seed)
            frame = pooled_and_within(r.normal(0, 1, 2 * n), r.normal(0, 1, 2 * n),
                                      group, n_boot=400)
            if frame.attrs['n_within_excluding_zero'] == 0:
                assert 'tracks performance within group' not in \
                    frame.attrs['interpretation']

    def test_returns_pooled_plus_one_row_per_group(self):
        rng = np.random.default_rng(4)
        n = 20
        frame = pooled_and_within(rng.normal(0, 1, 2 * n), rng.normal(0, 1, 2 * n),
                                  np.array(['AVGP'] * n + ['NVGP'] * n), n_boot=300)
        assert list(frame['label']) == ['pooled', 'within:AVGP', 'within:NVGP']

    def test_rejects_mismatched_group_length(self):
        with pytest.raises(ValueError, match='same length'):
            pooled_and_within([1.0, 2.0], [1.0, 2.0], ['AVGP'], n_boot=10)


class TestFeaturePerformanceTable:

    def test_covers_the_full_grid(self):
        rng = np.random.default_rng(0)
        n = 16
        group = np.array(['AVGP'] * n + ['NVGP'] * n)
        features = {'CON-SMN': rng.normal(0, 1, 2 * n),
                    'FPN-SMN': rng.normal(0, 1, 2 * n)}
        performance = {'accuracy': rng.normal(0, 1, 2 * n),
                       'rt': rng.normal(0, 1, 2 * n)}

        table = feature_performance_table(features, performance, group, n_boot=200)
        assert set(table['feature']) == {'CON-SMN', 'FPN-SMN'}
        assert set(table['measure']) == {'accuracy', 'rt'}
        # 2 features x 2 measures x (pooled + 2 groups)
        assert len(table) == 12
        assert 'no p-values' in table.attrs['note']
