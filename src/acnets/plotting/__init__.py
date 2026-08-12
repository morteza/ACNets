"""Figures for the manuscript.

`plot_tune_search_scores` needs `ray.tune.sklearn`, which lives in the optional
`tune` extra. Importing it eagerly here made the whole package -- and therefore
the specification-curve figure, which needs nothing but matplotlib -- fail to
import in a default install. It is exposed lazily instead, so the ImportError
arrives only if that specific function is asked for.
"""

from .specification_curve import parse_labels, specification_curve_figure

__all__ = ['parse_labels', 'plot_tune_search_scores',
           'specification_curve_figure']


def __getattr__(name):
    if name == 'plot_tune_search_scores':
        from .tune_search_plots import plot_tune_search_scores
        return plot_tune_search_scores
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
