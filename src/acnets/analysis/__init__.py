"""Analyses that relate the multiverse results to behaviour and to theory."""

from .brain_behavior import (
    Association,
    association,
    feature_performance_table,
    pooled_and_within,
)
from .multiscale import network_blocks, partial_correlation, regional_degree

__all__ = ['Association', 'association', 'feature_performance_table',
           'network_blocks', 'partial_correlation', 'pooled_and_within',
           'regional_degree']
