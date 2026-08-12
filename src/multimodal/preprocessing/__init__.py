"""Connectivity transformers, re-exported from `acnets.pipeline`.

These three transformers used to be forked into this package. The forks drifted
by two, eight and twenty-six lines respectively, in both directions: the copies
here recorded aggregation strategy and connectivity kind on the returned
dataset, which the originals did not, while `ConnectivityExtractor` here had
been narrowed back to nilearn's `ConnectivityMeasure` and so silently lost the
`chatterjee` and `transfer_entropy` statistics that the multiverse grid needs.

Both improvements now live in `acnets.pipeline` and this module is a re-export,
so `src/multimodal` and `src/acnets` cannot drift apart again. Import from
either path; they are the same objects.
"""

from acnets.pipeline import (
    ConnectivityAggregator,
    ConnectivityExtractor,
    TimeseriesAggregator,
)

__all__ = [
    'ConnectivityAggregator',
    'ConnectivityExtractor',
    'TimeseriesAggregator',
]
