"""Tests for the variance decomposition.

The construction tests inject accuracy that depends on exactly one dimension and
check the decomposition recovers it. This matters because an unbalanced grid can
easily produce a decomposition that attributes variance to whichever dimension
happens to correlate with the imbalance.
"""

import numpy as np
import pytest

from acnets.multiverse import SpecificationRegistry
from acnets.multiverse.variance import summarize, variance_decomposition


def _specs(n=None):
    """A subgrid orthogonal in every dimension it is decomposed over.

    Two restrictions, for two different confounds. The full grid confounds atlas
    with aggregation, because region-only atlases can never appear at network
    scale -- `.balanced` handles that one. It does *not* balance `pretraining`,
    which exists at network scale only, so this fixture pins pretraining to a
    single level as well.

    That second restriction used to be implicit: the registry default carried
    one pretraining level, so every grid was trivially balanced on it. Widening
    the default to match the manuscript's declared multiverse made the imbalance
    real, and `test_partial_never_exceeds_marginal_by_much` failed -- correctly,
    since partial R2 can exceed marginal R2 under suppression when predictors
    are correlated. See `test_unbalanced_pretraining_can_suppress`.
    """
    specs = [s for s in SpecificationRegistry().balanced
             if s.pretraining == 'none']
    return specs if n is None else specs[:n]


def _unbalanced_specs():
    return SpecificationRegistry().specifications


class TestVarianceDecomposition:

    def test_recovers_the_driving_dimension(self):
        """Accuracy driven only by the statistic; the statistic should dominate."""
        specs = _specs()
        offsets = {'correlation': 0.0, 'partial correlation': 0.2, 'covariance': 0.02,
                   'precision': 0.18, 'tangent': 0.1, 'chatterjee': 0.04,
                   'transfer_entropy': 0.06}
        acc = {s.spec_id: 0.5 + offsets[s.statistic] for s in specs}

        frame = variance_decomposition(acc, specs)
        assert frame['partial'].idxmax() == 'statistic'
        assert frame.loc['statistic', 'partial'] > 0.5

    def test_recovers_aggregation_when_it_drives(self):
        specs = _specs()
        offsets = {'region': 0.0, 'network': 0.22, 'random_network': 0.01}
        acc = {s.spec_id: 0.5 + offsets[s.aggregation] for s in specs}

        frame = variance_decomposition(acc, specs)
        assert frame['partial'].idxmax() == 'aggregation'

    def test_pure_noise_gives_small_components(self):
        rng = np.random.default_rng(0)
        specs = _specs()
        acc = {s.spec_id: float(rng.uniform(0.4, 0.7)) for s in specs}

        frame = variance_decomposition(acc, specs)
        # with no real structure, no single dimension should dominate
        assert frame['partial'].max() < 0.5

    def test_partial_never_exceeds_marginal_by_much(self):
        """Partial R2 is unique variance, so it cannot exceed marginal R2."""
        rng = np.random.default_rng(1)
        specs = _specs()
        acc = {s.spec_id: float(rng.uniform(0.4, 0.8)) for s in specs}

        frame = variance_decomposition(acc, specs)
        assert (frame['partial'] <= frame['marginal'] + 1e-9).all()

    def test_unbalanced_pretraining_can_suppress(self):
        """The invariant above is a property of orthogonality, not of the code.

        `.balanced` restores orthogonality between atlas, statistic, aggregation
        and classifier but explicitly not pretraining. Decomposing over an
        unbalanced dimension lets partial R2 exceed marginal R2 -- ordinary
        suppression between correlated predictors, not a bug -- and the warning
        attribute is how a caller is told.
        """
        rng = np.random.default_rng(1)
        specs = SpecificationRegistry().balanced
        assert len({s.pretraining for s in specs}) > 1, (
            'fixture assumes the default registry carries a pretraining axis')
        acc = {s.spec_id: float(rng.uniform(0.4, 0.8)) for s in specs}

        frame = variance_decomposition(acc, specs)
        assert (frame['partial'] > frame['marginal'] + 1e-9).any()

    def test_components_are_non_negative_and_bounded(self):
        rng = np.random.default_rng(2)
        specs = _specs()
        acc = {s.spec_id: float(rng.uniform(0.4, 0.8)) for s in specs}

        frame = variance_decomposition(acc, specs)
        assert (frame['partial'] >= 0).all()
        assert (frame['marginal'] <= 1.0 + 1e-9).all()

    def test_attrs_carry_full_model_and_caveat(self):
        specs = _specs()
        acc = {s.spec_id: 0.6 for s in specs}
        frame = variance_decomposition(acc, specs)
        assert 'r2_full' in frame.attrs
        assert 'descriptive only' in frame.attrs['note']

    def test_constant_accuracy_gives_zero_variance(self):
        specs = _specs()
        acc = {s.spec_id: 0.65 for s in specs}
        frame = variance_decomposition(acc, specs)
        assert frame['partial'].sum() == pytest.approx(0.0, abs=1e-8)
        assert frame['marginal'].sum() == pytest.approx(0.0, abs=1e-8)
        assert frame.attrs['r2_full'] == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize('value', [0.65, 0.1, 1.0, 0.0])
    @pytest.mark.parametrize('n', [4, 17, 210, 401])
    def test_constant_accuracy_is_flat_at_any_size(self, value, n):
        """Constant accuracy must decompose to nothing regardless of grid size.

        The mean of n identical floats is not always exactly that float, and how
        far off it lands depends on n. An exact `total == 0.0` guard therefore
        holds at some grid sizes and fails at others: it passed at 24
        specifications and produced a partial sum in the hundreds at 210.
        """
        specs = SpecificationRegistry().specifications[:n]
        acc = {s.spec_id: value for s in specs}
        frame = variance_decomposition(acc, specs)
        assert frame['partial'].sum() == pytest.approx(0.0, abs=1e-8)
        assert frame['marginal'].sum() == pytest.approx(0.0, abs=1e-8)

    def test_detects_the_atlas_aggregation_confound(self):
        """The full grid is unbalanced: region-only atlases never reach network
        scale, so atlas and aggregation cannot be separated. Must be flagged."""
        specs = _unbalanced_specs()
        acc = {s.spec_id: 0.6 for s in specs}
        frame = variance_decomposition(acc, specs)
        assert frame.attrs['balanced'] is False
        assert 'structurally' in frame.attrs['warning']

    def test_balanced_subgrid_is_not_flagged(self):
        specs = _specs()
        acc = {s.spec_id: 0.6 for s in specs}
        frame = variance_decomposition(acc, specs)
        assert frame.attrs['balanced'] is True
        assert 'warning' not in frame.attrs

    def test_requires_enough_specifications(self):
        specs = _specs(3)
        acc = {s.spec_id: 0.6 for s in specs}
        with pytest.raises(ValueError, match='at least four'):
            variance_decomposition(acc, specs)

    def test_ignores_unknown_spec_ids(self):
        specs = _specs()
        acc = {s.spec_id: 0.6 for s in specs}
        acc['not-a-real-id'] = 0.99
        frame = variance_decomposition(acc, specs)
        assert frame.attrs['n_specifications'] == len(specs)


class TestSummarize:

    def test_flattens_for_the_manuscript(self):
        specs = _specs()
        offsets = {'region': 0.0, 'network': 0.2, 'random_network': 0.0}
        acc = {s.spec_id: 0.5 + offsets[s.aggregation] for s in specs}

        out = summarize(variance_decomposition(acc, specs))
        assert out['ranked_by_partial'][0] == 'aggregation'
        assert set(out['partial']) == {'atlas', 'statistic', 'aggregation',
                                   'classifier', 'pretraining'}
        assert 0.0 <= out['r2_full'] <= 1.0
