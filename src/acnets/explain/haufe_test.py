"""Tests for the Haufe forward-model transform.

The critical test is test_recovers_pattern_under_correlated_noise: it constructs
data where the decoder weight is large on a pure-noise feature (because that
feature is needed to cancel noise) and checks the forward pattern correctly
assigns it near-zero importance. That is the exact failure mode Haufe et al.
(2014) describe, and it is why reading SVM weights directly is not safe.
"""

import numpy as np
import pytest

from acnets.explain.haufe import (
    average_patterns,
    haufe_pattern_series,
    haufe_transform,
)


class TestHaufeTransform:

    def test_identity_covariance_returns_weights(self):
        """With uncorrelated unit-variance features the transform is a no-op
        up to scale, so the ranking is unchanged."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (4000, 5))
        w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
        pattern = haufe_transform(X, w, normalize=False)
        # cosine similarity close to 1
        cos = pattern @ w / (np.linalg.norm(pattern) * np.linalg.norm(w))
        assert cos == pytest.approx(1.0, abs=0.05)

    def test_recovers_pattern_under_correlated_noise(self):
        """The failure mode that motivates the transform.

        f0 carries signal. f1 is pure noise, correlated with the noise in f0.
        An optimal decoder puts a large weight on f1 to subtract that noise, so
        the raw weight suggests f1 matters. The forward pattern should not.
        """
        rng = np.random.default_rng(0)
        n = 8000
        signal = rng.normal(0, 1, n)
        shared_noise = rng.normal(0, 1, n)

        f0 = signal + shared_noise
        f1 = shared_noise
        X = np.column_stack([f0, f1])

        # decoder that cancels the shared noise: f0 - f1 == signal
        w = np.array([1.0, -1.0])
        pattern = haufe_transform(X, w)

        assert abs(pattern[0]) > 5 * abs(pattern[1])

    def test_normalization_preserves_ranking(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (500, 8))
        w = rng.normal(0, 1, 8)
        raw = haufe_transform(X, w, normalize=False)
        scaled = haufe_transform(X, w, normalize=True)
        assert np.array_equal(np.argsort(np.abs(raw)), np.argsort(np.abs(scaled)))

    def test_sign_is_preserved(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (2000, 3))
        w = np.array([2.0, -2.0, 0.0])
        pattern = haufe_transform(X, w)
        assert pattern[0] > 0 and pattern[1] < 0

    def test_output_shape(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (50, 21))
        assert haufe_transform(X, rng.normal(0, 1, 21)).shape == (21,)

    def test_accepts_row_vector_weights(self):
        """sklearn LinearSVC.coef_ is (1, n_features)."""
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (100, 6))
        assert haufe_transform(X, rng.normal(0, 1, (1, 6))).shape == (6,)

    def test_rejects_mismatched_dimensions(self):
        X = np.zeros((10, 5))
        with pytest.raises(ValueError, match='features'):
            haufe_transform(X, np.zeros(4))

    def test_rejects_1d_X(self):
        with pytest.raises(ValueError, match='2d'):
            haufe_transform(np.zeros(10), np.zeros(10))

    def test_rejects_single_sample(self):
        with pytest.raises(ValueError, match='two samples'):
            haufe_transform(np.zeros((1, 5)), np.zeros(5))

    def test_zero_weights_give_zero_pattern(self):
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (100, 4))
        assert np.allclose(haufe_transform(X, np.zeros(4)), 0.0)


class TestHaufePatternSeries:

    def test_named_series(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 3))
        names = ['CON-SMN', 'FPN-SMN', 'CON-FPN']
        series = haufe_pattern_series(X, rng.normal(0, 1, 3), names)
        assert list(series.index) == names
        assert series.name == 'haufe_pattern'

    def test_rejects_wrong_name_count(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (100, 3))
        with pytest.raises(ValueError, match='feature names'):
            haufe_pattern_series(X, rng.normal(0, 1, 3), ['a', 'b'])


class TestAveragePatterns:

    def test_averages_across_folds(self):
        patterns = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        assert np.allclose(average_patterns(patterns), [2.0, 3.0])

    def test_flattens_row_vectors(self):
        patterns = [np.array([[1.0, 2.0]]), np.array([3.0, 4.0])]
        assert average_patterns(patterns).shape == (2,)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match='no patterns'):
            average_patterns([])
