from .connectivity_aggregator import ConnectivityAggregator
from .timeseries_aggregator import TimeseriesAggregator

__all__ = [
    'ConnectivityExtractor',
    'ConnectivityAggregator',
    'TimeseriesAggregator',
]

try:
    import nilearn
    from .connectivity_extractor import ConnectivityExtractor
    __all__.append('ConnectivityExtractor')
except ImportError:
    pass
