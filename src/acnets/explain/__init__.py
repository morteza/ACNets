"""Explainability for the multiverse classifiers.

Four explainability methods are used, deliberately chosen to fail differently:
SHAP, permutation importance, Haufe forward patterns for the linear models,
and attention weights from the deep models. Agreement across methods is stronger
evidence than any single one.
"""

from .haufe import average_patterns, haufe_pattern_series, haufe_transform

__all__ = ['average_patterns', 'haufe_pattern_series', 'haufe_transform']
