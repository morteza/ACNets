"""Tests for the section 3.2 feature blocks and the motion partial correlation."""

import numpy as np
import pytest
from scipy import stats

from acnets.analysis.multiscale import (
    network_blocks,
    partial_correlation,
    regional_degree,
)


class TestRegionalDegree:

    def test_matches_a_hand_computation(self):
        m = np.array([[[0.0, 1.0, -3.0],
                       [1.0, 0.0, 2.0],
                       [-3.0, 2.0, 0.0]]])
        # region 0: mean(|1|, |-3|) = 2; region 1: mean(1, 2) = 1.5; region 2: 2.5
        assert regional_degree(m)[0] == pytest.approx([2.0, 1.5, 2.5])

    def test_ignores_the_diagonal(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(4, 6, 6))
        m = (m + m.transpose(0, 2, 1)) / 2
        with_diagonal = m.copy()
        for k in range(4):
            np.fill_diagonal(with_diagonal[k], 99.0)
        assert regional_degree(m) == pytest.approx(regional_degree(with_diagonal))

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match='expected'):
            regional_degree(np.zeros((2, 3, 4)))


class TestNetworkBlocks:
    """The block definitions the paper's section 3.2 rests on."""

    def _toy(self):
        # 4 regions, 2 networks of 2; one subject with known entries
        m = np.array([[[0.0, 0.5, 0.1, 0.3],
                       [0.5, 0.0, 0.2, 0.4],
                       [0.1, 0.2, 0.0, 0.9],
                       [0.3, 0.4, 0.9, 0.0]]])
        return m, ['A', 'A', 'B', 'B'], ['A', 'B']

    def test_within_is_the_mean_over_region_pairs(self):
        m, assignment, networks = self._toy()
        within, _, names, _ = network_blocks(m, assignment, networks)
        assert names == ['A-A', 'B-B']
        assert within[0] == pytest.approx([0.5, 0.9])

    def test_between_is_the_mean_over_the_cross_block(self):
        m, assignment, networks = self._toy()
        _, between, _, names = network_blocks(m, assignment, networks)
        assert names == ['A-B']
        # mean of 0.1, 0.3, 0.2, 0.4
        assert between[0] == pytest.approx([0.25])

    def test_within_ignores_the_diagonal(self):
        """The defect this module exists to avoid: a constant diagonal.

        Correlation and partial-correlation matrices have a diagonal fixed at 1,
        so a within-network block that reads the diagonal is constant across
        subjects and carries no information.
        """
        m, assignment, networks = self._toy()
        poisoned = m.copy()
        np.fill_diagonal(poisoned[0], 1.0)
        a = network_blocks(m, assignment, networks)[0]
        b = network_blocks(poisoned, assignment, networks)[0]
        assert a[0] == pytest.approx(b[0])

    def test_blocks_vary_across_subjects_on_random_data(self):
        rng = np.random.default_rng(1)
        m = rng.normal(size=(20, 8, 8))
        m = (m + m.transpose(0, 2, 1)) / 2
        assignment = ['A'] * 4 + ['B'] * 4
        within, between, _, _ = network_blocks(m, assignment, ['A', 'B'])
        assert np.ptp(within, axis=0).min() > 0
        assert np.ptp(between, axis=0).min() > 0

    def test_rejects_a_singleton_network(self):
        m, _, _ = self._toy()
        with pytest.raises(ValueError, match='fewer than 2'):
            network_blocks(m, ['A', 'B', 'B', 'B'], ['A', 'B'])

    def test_rejects_mismatched_assignment_length(self):
        m, _, networks = self._toy()
        with pytest.raises(ValueError, match='assignments'):
            network_blocks(m, ['A', 'A'], networks)


class TestPartialCorrelation:

    def test_matches_the_closed_form(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=200)
        x = 0.7 * z + rng.normal(size=200)
        y = 0.5 * z + 0.3 * x + rng.normal(size=200)

        r, _ = partial_correlation(x, y, z)
        rxy = stats.pearsonr(x, y)[0]
        rxz = stats.pearsonr(x, z)[0]
        ryz = stats.pearsonr(y, z)[0]
        expected = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        assert r == pytest.approx(expected)

    def test_a_purely_confounded_association_goes_to_zero(self):
        """x and y correlate only because both depend on z."""
        rng = np.random.default_rng(2)
        z = rng.normal(size=500)
        x = 2.0 * z + 0.1 * rng.normal(size=500)
        y = -3.0 * z + 0.1 * rng.normal(size=500)

        assert abs(stats.pearsonr(x, y)[0]) > 0.9      # strong, and spurious
        r, p = partial_correlation(x, y, z)
        assert abs(r) < 0.15
        assert p > 0.001

    def test_a_genuine_association_survives(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=300)
        x = z + rng.normal(size=300)
        y = 0.8 * x + rng.normal(size=300)
        r, p = partial_correlation(x, y, z)
        assert r > 0.4
        assert p < 1e-6

    def test_rejects_ragged_input(self):
        with pytest.raises(ValueError, match='lengths differ'):
            partial_correlation([1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4])
