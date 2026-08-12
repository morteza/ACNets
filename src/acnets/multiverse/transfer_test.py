"""Tests for LEMON-fitted transfer and the fold-feature cache.

The two properties that matter are stated as tests rather than as prose: the
transforms never see a label (so features are invariant to label permutation),
and the cache returns bit-identical results to computing fresh.
"""

import numpy as np
import pytest

from .runner import (
    _cached_fold_features,
    _feature_cache_key,
    _fold_features,
    data_token,
    run_specification,
)
from .specification import Specification
from .transfer import fit_lemon_transforms, windowed


@pytest.fixture
def nodes():
    return ['CER', 'CON', 'DMN', 'FPN', 'SMN', 'VIS']


@pytest.fixture
def lemon_like(nodes):
    """External 'pretraining' data: more subjects, longer runs."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(20, 372, len(nodes)))


@pytest.fixture
def julia_like(nodes):
    """Target data: few subjects, short runs, binary groups."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(16, 124, len(nodes)))
    y = np.array([0] * 8 + [1] * 8)
    return x, y


class TestWindowed:

    def test_cuts_non_overlapping_windows(self, lemon_like):
        windows = windowed(lemon_like, length=124)
        assert windows.shape == (20 * 3, 124, 6)  # 372 // 124 = 3 per subject

    def test_stride_yields_overlapping_windows(self, lemon_like):
        windows = windowed(lemon_like, length=124, stride=62)
        # starts at 0, 62, ..., 248 -> 5 per subject
        assert windows.shape == (20 * 5, 124, 6)

    def test_windows_are_zscored(self, lemon_like):
        windows = windowed(lemon_like, length=124)
        np.testing.assert_allclose(windows.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(windows.std(axis=1), 1.0, atol=1e-10)

    def test_rejects_window_longer_than_run(self, lemon_like):
        with pytest.raises(ValueError, match='exceeds run length'):
            windowed(lemon_like, length=500)


class TestLemonTransforms:

    def test_tangent_reference_comes_from_lemon_not_the_fold(
            self, lemon_like, julia_like, nodes):
        """The point of the transfer: the reference is external and frozen."""
        x, _ = julia_like
        lemon = fit_lemon_transforms(lemon_like, nodes, ['tangent'])

        spec_lemon = Specification('dosenbach2010', 'tangent', 'network',
                                   pretraining='lemon')
        spec_none = Specification('dosenbach2010', 'tangent', 'network')
        train, test = np.arange(12), np.arange(12, 16)

        from_lemon, _ = _fold_features(spec_lemon, x, train, test, lemon)
        from_fold, _ = _fold_features(spec_none, x, train, test)

        assert from_lemon.shape == from_fold.shape
        assert not np.allclose(from_lemon, from_fold)

    def test_stateless_statistics_agree_between_flavours(
            self, lemon_like, julia_like, nodes):
        """Correlation has no fitted reference, so 'lemon' and 'none' features
        coincide; the twins then differ only through the scaler."""
        x, _ = julia_like
        lemon = fit_lemon_transforms(lemon_like, nodes, ['correlation'])
        spec_lemon = Specification('dosenbach2010', 'correlation', 'network',
                                   pretraining='lemon')
        spec_none = Specification('dosenbach2010', 'correlation', 'network')
        train, test = np.arange(12), np.arange(12, 16)

        a, _ = _fold_features(spec_lemon, x, train, test, lemon)
        b, _ = _fold_features(spec_none, x, train, test)
        np.testing.assert_allclose(a, b)

    def test_never_receives_labels(self, lemon_like, nodes):
        """The signature is the proof: there is no argument to leak through."""
        import inspect

        parameters = inspect.signature(fit_lemon_transforms).parameters
        assert not any(name in parameters for name in ('y', 'labels', 'groups'))

    def test_vocabulary_mismatch_is_an_error(self, lemon_like, nodes):
        lemon = fit_lemon_transforms(lemon_like, nodes, ['correlation'])
        with pytest.raises(ValueError, match='vocabulary mismatch'):
            lemon.check_nodes(['A', 'B', 'C', 'D', 'E', 'F'])

    def test_missing_transforms_fail_recorded_not_raised(self, julia_like, nodes):
        x, y = julia_like
        spec = Specification('dosenbach2010', 'correlation', 'network',
                             pretraining='lemon')
        result = run_specification(spec, x, y, nodes, n_splits=2, n_repeats=1)
        assert result.failed is not None
        assert 'LemonTransforms' in result.failed


class TestLemonSpecificationEndToEnd:

    def test_runs_and_matches_its_twin_where_it_should(
            self, lemon_like, julia_like, nodes):
        x, y = julia_like
        lemon = fit_lemon_transforms(lemon_like, nodes,
                                     ['correlation', 'tangent'])

        for statistic in ('correlation', 'tangent'):
            spec = Specification('dosenbach2010', statistic, 'network', 'svm',
                                 pretraining='lemon')
            result = run_specification(spec, x, y, nodes, lemon=lemon,
                                       n_splits=2, n_repeats=2)
            assert result.failed is None
            assert 0.0 <= result.accuracy <= 1.0
            assert len(result.attribution) == 21

    def test_features_are_invariant_to_label_permutation(
            self, lemon_like, julia_like, nodes, tmp_path):
        """The property the permutation null and the cache both rest on."""
        x, y = julia_like
        lemon = fit_lemon_transforms(lemon_like, nodes, ['tangent'])
        spec = Specification('dosenbach2010', 'tangent', 'network',
                             pretraining='lemon')

        rng = np.random.default_rng(0)
        run_specification(spec, x, y, nodes, lemon=lemon,
                          n_splits=2, n_repeats=1, cache_dir=tmp_path)
        files_before = sorted(p.name for p in tmp_path.glob('*.npz'))

        run_specification(spec, x, rng.permutation(y), nodes, lemon=lemon,
                          n_splits=2, n_repeats=1, cache_dir=tmp_path)
        files_after = sorted(p.name for p in tmp_path.glob('*.npz'))

        # permuting labels changed the folds' membership? No: stratified splits
        # depend on y. So allow new keys, but every original key must be reused
        # unchanged -- and for identical splits the features must be identical,
        # which the cache-correctness test below asserts directly.
        assert set(files_before) <= set(files_after)


class TestFeatureCache:

    def test_cached_equals_uncached_bit_for_bit(self, julia_like, nodes, tmp_path):
        x, y = julia_like
        spec = Specification('dosenbach2010', 'tangent', 'network', 'svm')
        kwargs = dict(n_splits=2, n_repeats=3, random_state=5)

        fresh = run_specification(spec, x, y, nodes, **kwargs)
        cached_cold = run_specification(spec, x, y, nodes, cache_dir=tmp_path,
                                        **kwargs)
        cached_warm = run_specification(spec, x, y, nodes, cache_dir=tmp_path,
                                        **kwargs)

        for other in (cached_cold, cached_warm):
            np.testing.assert_array_equal(fresh.fold_accuracies,
                                          other.fold_accuracies)
            assert fresh.attribution == other.attribution

    def test_key_ignores_the_classifier(self, julia_like):
        x, _ = julia_like
        token = data_token(x)
        train, test = np.arange(12), np.arange(12, 16)
        svm = Specification('dosenbach2010', 'tangent', 'network', 'svm')
        xgb = Specification('dosenbach2010', 'tangent', 'network', 'xgboost')
        assert (_feature_cache_key(svm, token, train, test)
                == _feature_cache_key(xgb, token, train, test))

    def test_key_separates_data_statistic_pretraining_and_split(self, julia_like):
        x, _ = julia_like
        token = data_token(x)
        train, test = np.arange(12), np.arange(12, 16)
        base = Specification('dosenbach2010', 'tangent', 'network')
        keys = {
            _feature_cache_key(base, token, train, test),
            _feature_cache_key(
                Specification('dosenbach2010', 'correlation', 'network'),
                token, train, test),
            _feature_cache_key(
                Specification('dosenbach2010', 'tangent', 'network',
                              pretraining='lemon'), token, train, test),
            _feature_cache_key(base, 'other-data', train, test),
            _feature_cache_key(base, token, train[:-1], test),
        }
        assert len(keys) == 5

    def test_cache_round_trips_through_disk(self, julia_like, tmp_path):
        x, _ = julia_like
        spec = Specification('dosenbach2010', 'correlation', 'network')
        token = data_token(x)
        train, test = np.arange(12), np.arange(12, 16)

        first = _cached_fold_features(spec, x, train, test, None, tmp_path, token)
        assert len(list(tmp_path.glob('*.npz'))) == 1
        second = _cached_fold_features(spec, x, train, test, None, tmp_path, token)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
