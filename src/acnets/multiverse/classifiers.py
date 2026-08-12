"""Classifier families and their search spaces, with hyperparameters tuned in-fold.

Two problems this fixes.

**The paper claimed tuning that was not happening.** Methods section 6.5 said
"hyperparameters were tuned using only the training data", but
`runner._make_classifier` returned fixed values -- `C=1.0`, `max_depth=3`,
`learning_rate=0.1` -- chosen once and never varied. The claim was false. Either
the sentence or the code had to change; tuning is the better fix, because
"you didn't tune" is otherwise a free objection for a reviewer.

**Three model families is a thin sample of model space.** At 32 participants and
21 features the models that usually win are heavily regularised linear ones, and
the grid contained exactly one (an L1 SVM at fixed C). Shrinkage LDA, ridge and
regularised logistic regression are the natural occupants of this regime and
were all missing.

**Tuning here does not inflate the estimate, and that is the whole point.** The
search runs on an inner cross-validation *inside each training fold*; the outer
test fold is never seen by the search. This is the standard nested protocol and
it is what makes a tuned accuracy honest. `classifiers_test.py` asserts it the
only way that really works -- by tuning against permuted labels and requiring the
result to stay at chance. A leak would show up there as above-chance accuracy on
data with no signal in it.

What tuning is *not* is free accuracy. It usually moves small-sample results very
little, and every family added widens the family-wise null the result must clear
(`acnets.multiverse.inference`). Both effects are measured rather than assumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

# Families worth having at n = 32. The bias is deliberately toward strong
# regularisation and toward estimators that degrade gracefully when the feature
# count approaches the sample size; `qda` and `knn` are included as the two that
# should *not* cope, so the comparison has a floor that is not chance.
CLASSIFIERS: dict[str, dict[str, Any]] = {
    'svm': {
        'estimator': lambda seed: LinearSVC(dual='auto', max_iter=20000,
                                            random_state=seed),
        'space': {'C': np.logspace(-3, 2, 12)},
    },
    'svm-rbf': {
        'estimator': lambda seed: SVC(kernel='rbf', random_state=seed),
        'space': {'C': np.logspace(-2, 3, 10),
                  'gamma': np.logspace(-4, 0, 8)},
    },
    'logistic': {
        'estimator': lambda seed: LogisticRegression(max_iter=5000,
                                                     random_state=seed),
        'space': {'C': np.logspace(-3, 2, 12)},
    },
    'logistic-en': {
        # sklearn >= 1.8 deprecates `penalty`; the mix is set through l1_ratio
        # alone, where 0 is pure L2 and 1 is pure L1
        'estimator': lambda seed: LogisticRegression(
            solver='saga', max_iter=8000, random_state=seed),
        'space': {'C': np.logspace(-3, 1, 8),
                  'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
    },
    'ridge': {
        'estimator': lambda seed: RidgeClassifier(random_state=seed),
        'space': {'alpha': np.logspace(-2, 3, 12)},
    },
    'lda': {
        # lsqr + shrinkage is the small-sample discriminant: it regularises the
        # covariance estimate, which is exactly what fails at p close to n
        'estimator': lambda seed: LinearDiscriminantAnalysis(solver='lsqr'),
        'space': {'shrinkage': [*np.linspace(0.0, 1.0, 11), 'auto']},
    },
    'qda': {
        'estimator': lambda seed: QuadraticDiscriminantAnalysis(),
        'space': {'reg_param': np.linspace(0.0, 0.9, 10)},
    },
    'gnb': {
        'estimator': lambda seed: GaussianNB(),
        'space': {'var_smoothing': np.logspace(-11, -3, 9)},
    },
    'knn': {
        'estimator': lambda seed: KNeighborsClassifier(),
        'space': {'n_neighbors': [3, 5, 7, 9, 11],
                  'weights': ['uniform', 'distance']},
    },
    'random-forest': {
        'estimator': lambda seed: RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=1),
        'space': {'max_depth': [2, 3, 4, None],
                  'max_features': ['sqrt', 'log2', 0.5],
                  'min_samples_leaf': [1, 2, 4]},
    },
    'extra-trees': {
        'estimator': lambda seed: ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=1),
        'space': {'max_depth': [2, 3, 4, None],
                  'max_features': ['sqrt', 'log2', 0.5],
                  'min_samples_leaf': [1, 2, 4]},
    },
    'xgboost': {
        'estimator': None,          # imported lazily; see _estimator
        'space': {'max_depth': [2, 3, 4],
                  'learning_rate': [0.01, 0.05, 0.1, 0.2],
                  'n_estimators': [100, 200, 400],
                  'subsample': [0.6, 0.8, 1.0],
                  'colsample_bytree': [0.6, 0.8, 1.0],
                  'reg_lambda': [0.1, 1.0, 10.0]},
    },
    'mlp': {
        'estimator': None,          # torch, imported lazily
        'space': {'hidden': [8, 16, 32],
                  'lr': [1e-3, 1e-2],
                  'epochs': [50, 100, 200]},
    },
}

DEFAULT_CLASSIFIERS = ('svm', 'xgboost', 'mlp')


def _estimator(name: str, seed: int):
    """Build an untuned estimator, importing optional backends lazily."""
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=1, eval_metric='logloss')
    if name == 'mlp':
        # xgboost must bind OpenMP before torch, or a later XGBoost fit
        # segfaults in the same process (acnets._openmp)
        from .._openmp import preload
        preload()
        from ..deep.torch_classifier import TorchMLPClassifier
        return TorchMLPClassifier(random_state=seed)
    if name not in CLASSIFIERS:
        raise ValueError(f'unknown classifier: {name!r}')
    return CLASSIFIERS[name]['estimator'](seed)


def make_classifier(name: str, random_state: int = 0, tuned: bool = False,
                    n_iter: int = 20, inner_splits: int = 3,
                    n_jobs: int = 1):
    """A classifier, optionally wrapping an in-fold hyperparameter search.

    Parameters
    ----------
    tuned
        When False, returns the estimator with fixed defaults -- the historical
        behaviour, kept so existing results stay reproducible. When True, wraps
        it in a `RandomizedSearchCV`, so calling `.fit(x_train, y_train)`
        performs the search on an inner split of the training data only.
    n_iter
        Hyperparameter configurations sampled. The spaces here are small enough
        that 20 covers most of them.
    inner_splits
        Folds of the inner cross-validation. Three is the practical floor at 24
        training participants; more leaves too few per inner fold.

    The search is over the *training* argument it receives and nothing else,
    which is what makes the outer accuracy an honest estimate.
    """
    estimator = _estimator(name, random_state)
    if not tuned:
        return estimator

    space = CLASSIFIERS[name]['space']
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                            random_state=random_state)
    return RandomizedSearchCV(
        estimator, space, n_iter=n_iter, cv=inner, scoring='accuracy',
        random_state=random_state, n_jobs=n_jobs, refit=True,
        error_score=0.5)          # a failed configuration scores chance, not NaN


def available(include_optional: bool = True) -> list[str]:
    """Classifier names that can actually be constructed in this environment."""
    names = []
    for name in CLASSIFIERS:
        if not include_optional and name in ('xgboost', 'mlp'):
            continue
        try:
            _estimator(name, 0)
        except ImportError:
            continue
        names.append(name)
    return names
