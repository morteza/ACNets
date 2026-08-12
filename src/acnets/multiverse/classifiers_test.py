"""Tests for the classifier registry and in-fold hyperparameter search.

The load-bearing test is `test_tuning_on_permuted_labels_stays_at_chance`. Every
other property here is bookkeeping; that one is the difference between a tuned
accuracy that means something and one that does not.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .classifiers import CLASSIFIERS, available, make_classifier
from .runner import fixed_splits


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8))
    y = (x[:, 0] + 0.4 * x[:, 1] + 0.3 * rng.normal(size=60) > 0).astype(int)
    return x, y


class TestRegistry:

    def test_every_family_builds_and_fits(self, separable):
        x, y = separable
        for name in available():
            model = make_classifier(name, random_state=0)
            model.fit(x, y)
            assert model.predict(x).shape == (60,), name

    def test_every_family_has_a_non_empty_search_space(self):
        for name, entry in CLASSIFIERS.items():
            assert entry['space'], name
            for parameter, values in entry['space'].items():
                assert len(values) > 0, f'{name}.{parameter}'

    def test_search_spaces_name_real_parameters(self, separable):
        """A typo in a space would only surface at fit time, deep in a run."""
        for name in available():
            estimator = make_classifier(name, random_state=0)
            valid = set(estimator.get_params())
            for parameter in CLASSIFIERS[name]['space']:
                assert parameter in valid, f'{name}: {parameter!r} not a parameter'

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match='unknown classifier'):
            make_classifier('does-not-exist')

    def test_untuned_is_the_default(self):
        assert not isinstance(make_classifier('ridge'), RandomizedSearchCV)
        assert isinstance(make_classifier('ridge', tuned=True), RandomizedSearchCV)


class TestTuning:

    def test_tuned_estimator_searches_and_refits(self, separable):
        x, y = separable
        model = make_classifier('ridge', tuned=True, n_iter=5)
        model.fit(x, y)
        assert hasattr(model, 'best_params_')
        assert 'alpha' in model.best_params_
        assert model.predict(x).shape == (60,)

    def test_search_uses_an_inner_split_of_what_it_is_given(self, separable):
        x, y = separable
        model = make_classifier('ridge', tuned=True, n_iter=3)
        assert isinstance(model.cv, StratifiedKFold)
        model.fit(x[:40], y[:40])
        # n_splits_ folds over 40 rows: the search never saw rows 40..59
        assert model.n_splits_ == 3

    def test_tuning_on_permuted_labels_stays_at_chance(self):
        """The leakage test: tuning cannot manufacture signal that is not there.

        If the inner search could see the outer test fold, tuned accuracy on
        randomly permuted labels would climb above 0.5. It must not.
        """
        rng = np.random.default_rng(0)
        x = rng.normal(size=(32, 21))
        y = rng.permutation(np.array([0] * 16 + [1] * 16))
        splits = fixed_splits(32, n_splits=4, n_repeats=5, random_state=0)

        folds = []
        for train, test in splits:
            scaler = StandardScaler().fit(x[train])
            model = make_classifier('ridge', tuned=True, n_iter=6, random_state=0)
            model.fit(scaler.transform(x[train]), y[train])
            folds.append((model.predict(scaler.transform(x[test])) == y[test]).mean())

        accuracy = float(np.mean(folds))
        assert 0.35 < accuracy < 0.65, f'tuned accuracy {accuracy} on random labels'

    def test_tuned_model_is_clonable(self):
        model = make_classifier('lda', tuned=True, n_iter=4)
        assert clone(model).get_params()['n_iter'] == 4

    def test_a_failing_configuration_scores_chance_not_nan(self, separable):
        """error_score=0.5 keeps one bad draw from poisoning the whole search."""
        model = make_classifier('qda', tuned=True, n_iter=4)
        assert model.error_score == 0.5


class TestAvailable:

    def test_lists_only_constructible_families(self):
        names = available()
        assert 'ridge' in names and 'lda' in names
        for name in names:
            make_classifier(name)

    def test_can_exclude_optional_backends(self):
        names = available(include_optional=False)
        assert 'xgboost' not in names
        assert 'mlp' not in names
        assert 'svm' in names
