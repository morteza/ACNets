"""Tests for the sklearn-compatible MLP and its place in the multiverse."""

import numpy as np
import pytest

from .torch_classifier import TorchMLPClassifier


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, 21))
    y = (x[:, 3] + 0.5 * x[:, 7] > 0).astype(int)
    return x, y


class TestTorchMLPClassifier:

    def test_fits_and_predicts_separable_data(self, separable):
        x, y = separable
        model = TorchMLPClassifier(epochs=150).fit(x, y)
        assert (model.predict(x) == y).mean() > 0.9

    def test_deterministic_under_its_seed(self, separable):
        x, y = separable
        a = TorchMLPClassifier(random_state=7).fit(x, y).predict_proba(x)
        b = TorchMLPClassifier(random_state=7).fit(x, y).predict_proba(x)
        np.testing.assert_array_equal(a, b)

    def test_attribution_ranks_the_informative_features_first(self, separable):
        x, y = separable
        model = TorchMLPClassifier(epochs=150).fit(x, y)
        attribution = model.attribution(x)
        assert attribution.shape == (21,)
        assert set(np.argsort(attribution)[-2:]) == {3, 7}

    def test_runs_inside_the_multiverse_runner(self, separable):
        """The point of the wrapper: an 'mlp' specification is just a spec."""
        from acnets.multiverse import Specification, run_specification

        rng = np.random.default_rng(1)
        nodes = ['CER', 'CON', 'DMN', 'FPN', 'SMN', 'VIS']
        timeseries = rng.normal(size=(16, 60, 6))
        labels = np.array([0] * 8 + [1] * 8)

        spec = Specification('dosenbach2010', 'correlation', 'network', 'mlp')
        result = run_specification(spec, timeseries, labels, nodes,
                                   n_splits=2, n_repeats=1)
        assert result.failed is None
        assert len(result.attribution) == 21
        assert all(np.isfinite(v) for v in result.attribution.values())


class TestSklearnCompatibility:
    """The classifier must compose, not merely duck-type.

    `runner._make_classifier` only calls fit/predict, so duck-typing passed
    every test while `Pipeline([..., ('clf', TorchMLPClassifier())])` raised
    AttributeError on `__sklearn_tags__` under sklearn >= 1.6. Anything that
    puts the model inside a Pipeline, a search, or a cross_val_* helper hit it.
    """

    def test_works_inside_a_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 6))
        y = (x[:, 0] > 0).astype(int)

        pipeline = Pipeline([('scale', StandardScaler()),
                             ('clf', TorchMLPClassifier(epochs=30))])
        pipeline.fit(x, y)
        assert pipeline.predict(x).shape == (40,)

    def test_works_with_cross_val_score(self):
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 6))
        y = (x[:, 0] > 0).astype(int)
        scores = cross_val_score(TorchMLPClassifier(epochs=30), x, y, cv=3)
        assert scores.shape == (3,)

    def test_exposes_constructor_params(self):
        params = TorchMLPClassifier(hidden=32).get_params()
        assert params['hidden'] == 32
        assert 'epochs' in params and 'lr' in params

    def test_is_clonable(self):
        from sklearn.base import clone

        original = TorchMLPClassifier(hidden=7, epochs=3)
        assert clone(original).get_params() == original.get_params()
