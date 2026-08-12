"""Haufe transform: from decoding weights to interpretable forward patterns.

Haufe et al. (2014, NeuroImage 87, 96-110) show that the weights of a backward
(decoding) model such as an SVM or LASSO cannot be read as statements about the
underlying brain processes, because the extraction filter depends on every noise
component in the data. A feature can carry a large weight purely because it
helps suppress noise in another feature.

Since the interpretive claim of this project rests entirely on reading a
decoding model, this transform is not optional. For a linear model with weight
vector w and data covariance Sigma, the corresponding forward pattern is

    A = Sigma w / (w' Sigma w)

which is what the model implies about how the discriminative source is
expressed in the measured features.
"""

from __future__ import annotations

import numpy as np


def haufe_transform(X: np.ndarray,
                    weights: np.ndarray,
                    normalize: bool = True) -> np.ndarray:
    """Convert linear decoder weights into a forward activation pattern.

    Parameters
    ----------
    X
        Feature matrix, shape (n_samples, n_features). Should be the same
        representation the classifier was fitted on, after any scaling.
    weights
        Decoder weight vector, shape (n_features,). For scikit-learn's
        LinearSVC this is `clf.coef_.ravel()`.
    normalize
        Divide by the variance of the projected scores, giving the pattern in
        the units of the original features. When False the returned pattern is
        proportional to Sigma @ w, which preserves the ranking but not the scale.

    Returns
    -------
    ndarray of shape (n_features,)
        Signed forward pattern. Sign indicates the direction of the association
        with the positive class; magnitude indicates how strongly the
        discriminative source is expressed in that feature.

    Notes
    -----
    Rank the pattern by magnitude, not by signed value, when comparing across
    specifications: a strongly negative expression is as informative as a
    strongly positive one.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError(f'X must be 2d, got shape {X.shape}')
    if X.shape[1] != weights.size:
        raise ValueError(
            f'X has {X.shape[1]} features but weights has {weights.size}')
    if X.shape[0] < 2:
        raise ValueError('need at least two samples to estimate covariance')

    X_centred = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X_centred, rowvar=False)
    cov = np.atleast_2d(cov)

    pattern = cov @ weights

    if normalize:
        scores = X_centred @ weights
        score_variance = float(np.var(scores, ddof=1))
        if score_variance > 0.0:
            pattern = pattern / score_variance

    return pattern


def haufe_pattern_series(X: np.ndarray,
                         weights: np.ndarray,
                         feature_names: list[str],
                         normalize: bool = True):
    """haufe_transform as a named pandas Series, for the agreement module.

    Kept separate so haufe_transform stays a pure numpy function.
    """
    import pandas as pd

    pattern = haufe_transform(X, weights, normalize=normalize)
    if len(feature_names) != pattern.size:
        raise ValueError(
            f'{len(feature_names)} feature names for {pattern.size} features')
    return pd.Series(pattern, index=feature_names, name='haufe_pattern')


def average_patterns(patterns: list[np.ndarray]) -> np.ndarray:
    """Average forward patterns across cross-validation folds.

    Averaging patterns is meaningful in a way that averaging raw decoder weights
    is not, because forward patterns live in a fixed feature space and share a
    common interpretation across folds.
    """
    if not patterns:
        raise ValueError('no patterns to average')
    stacked = np.vstack([np.asarray(p, dtype=float).ravel() for p in patterns])
    return stacked.mean(axis=0)
