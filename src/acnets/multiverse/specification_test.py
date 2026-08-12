"""Tests for Specification, focused on the asymmetry bug it guards against."""

import pytest

from acnets.multiverse import ASYMMETRIC_STATISTICS, Specification


class TestAsymmetry:
    """Transfer entropy and Chatterjee xi are directed; the other five are not.

    Vectorising by upper triangle is lossless only for symmetric matrices, so
    conflating the two silently discards the directional information that
    motivated including these statistics at all.
    """

    def test_directed_statistics_flagged(self):
        for stat in ('transfer_entropy', 'chatterjee'):
            spec = Specification('dosenbach2010', stat, 'network')
            assert spec.is_asymmetric, stat

    def test_symmetric_statistics_not_flagged(self):
        for stat in ('correlation', 'partial correlation', 'covariance',
                     'precision', 'tangent'):
            spec = Specification('dosenbach2010', stat, 'network')
            assert not spec.is_asymmetric, stat

    def test_asymmetric_set_matches_property(self):
        for stat in ASYMMETRIC_STATISTICS:
            assert Specification('dosenbach2010', stat, 'network').is_asymmetric

    def test_feature_count_differs_by_symmetry(self):
        """6 networks: 21 features if symmetric, 36 if directed."""
        sym = Specification('dosenbach2010', 'partial correlation', 'network')
        asym = Specification('dosenbach2010', 'transfer_entropy', 'network')
        assert sym.n_features(6) == 21
        assert asym.n_features(6) == 36
        assert asym.n_features(6) > sym.n_features(6)

    def test_feature_count_rejects_bad_node_count(self):
        spec = Specification('dosenbach2010', 'correlation', 'network')
        with pytest.raises(ValueError, match='positive'):
            spec.n_features(0)


class TestSpecIdentity:

    def test_spec_id_is_stable_and_content_addressed(self):
        a = Specification('dosenbach2010', 'precision', 'network', 'svm')
        b = Specification('dosenbach2010', 'precision', 'network', 'svm')
        c = Specification('dosenbach2010', 'precision', 'network', 'xgboost')
        assert a.spec_id == b.spec_id
        assert a.spec_id != c.spec_id
        assert len(a.spec_id) == 12

    def test_roundtrip_through_dict(self):
        spec = Specification('gordon2014', 'tangent', 'random_network', 'xgboost')
        assert Specification.from_dict(spec.to_dict()) == spec

    def test_control_flag(self):
        assert Specification('gordon2014', 'correlation', 'random_network').is_control
        assert not Specification('gordon2014', 'correlation', 'network').is_control

    def test_rejects_unknown_aggregation(self):
        with pytest.raises(ValueError, match='unknown aggregation'):
            Specification('dosenbach2010', 'correlation', 'bogus')


class TestPretrainingDegeneracy:
    """A `lemon` specification must differ from its `none` twin, or not exist.

    LEMON pretraining changes exactly two fitted objects: the connectivity
    measure, which only carries state for `tangent`, and the feature scaler,
    which only a scale-sensitive classifier can see. A tree ensemble splits on
    order statistics, so `precision + xgboost + lemon` computes bit-identically
    to `precision + xgboost`. Emitting both put five exact duplicates in the
    54-cell grid, which overstated the number of analysis paths and handed the
    attribution-agreement mean five free 1.0s.
    """

    def test_stateless_statistic_with_tree_is_rejected(self):
        for statistic in ('correlation', 'partial correlation', 'covariance',
                          'precision', 'chatterjee'):
            spec = Specification('dosenbach2010', statistic, 'network',
                                 'xgboost', 'lemon')
            ok, reason = spec.is_valid()
            assert not ok, statistic
            assert 'no-op' in reason

    def test_tangent_with_tree_is_kept(self):
        # the measure carries a fitted reference, so this one is substantive
        spec = Specification('dosenbach2010', 'tangent', 'network',
                             'xgboost', 'lemon')
        assert spec.is_valid()[0]

    def test_scale_sensitive_classifiers_are_kept(self):
        for classifier in ('svm', 'mlp'):
            spec = Specification('dosenbach2010', 'precision', 'network',
                                 classifier, 'lemon')
            assert spec.is_valid()[0], classifier

    def test_no_duplicate_computations_in_the_grid(self):
        from acnets.multiverse import SpecificationRegistry

        registry = SpecificationRegistry(
            atlases=('dosenbach2010',),
            aggregations=('network', 'random_network'),
            classifiers=('svm', 'xgboost', 'mlp'),
            pretrainings=('none', 'lemon'))

        # what actually determines the computation, ignoring the axis label
        def computation(spec):
            transfers = (spec.pretraining == 'lemon'
                         and (spec.statistic in ('tangent',)
                              or spec.classifier in ('svm', 'mlp')))
            return (spec.atlas, spec.statistic, spec.aggregation,
                    spec.classifier, transfers)

        signatures = [computation(s) for s in registry]
        assert len(signatures) == len(set(signatures))
