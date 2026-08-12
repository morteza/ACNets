"""Multiverse analysis: enumerate analysis paths, then reason over the set.

The design is append-only with projections. Specifications are enumerated by a
declarative registry, each (specification, run, fold) result is an immutable
record, and every figure in the paper is a projection computed on read. A
reviewer asking "what if you had excluded transfer entropy" is answered by
re-projecting rather than re-running.
"""

from .agreement import (
    AgreementResult,
    accuracy_agreement_baseline,
    agreement_matrix,
    agreement_vs_threshold,
    excess_agreement,
    kendall_agreement,
    pairwise_accuracy_agreement,
    random_agreement_baseline,
    rank_biased_overlap,
    selection_frequency,
    stability_comparison,
    stability_vs_performance,
    to_ranking,
)
from .inference import (
    AccuracyReport,
    Interval,
    MaxStatResult,
    accuracy_interval,
    accuracy_report,
    corrected_resampled_t,
    max_stat_permutation,
    permutation_pvalue,
    specification_curve,
)
from .registry import SpecificationRegistry, network_atlases
from .runner import (
    RunOutcome,
    SpecificationResult,
    edge_names,
    fixed_splits,
    run_specification,
    run_specifications,
    vectorize,
)
from .specification import (
    ALL_ATLASES,
    ALL_STATISTICS,
    ASYMMETRIC_STATISTICS,
    EXTRA_STATISTICS,
    NETWORK_ATLASES,
    NILEARN_STATISTICS,
    REGION_ONLY_ATLASES,
    Specification,
)
from .transfer import LemonTransforms, fit_lemon_transforms, windowed
from .variance import summarize, variance_decomposition

__all__ = [
    'ALL_ATLASES',
    'ALL_STATISTICS',
    'ASYMMETRIC_STATISTICS',
    'EXTRA_STATISTICS',
    'NETWORK_ATLASES',
    'NILEARN_STATISTICS',
    'REGION_ONLY_ATLASES',
    'AccuracyReport',
    'AgreementResult',
    'Interval',
    'LemonTransforms',
    'MaxStatResult',
    'RunOutcome',
    'Specification',
    'SpecificationRegistry',
    'SpecificationResult',
    'accuracy_agreement_baseline',
    'accuracy_interval',
    'accuracy_report',
    'agreement_matrix',
    'agreement_vs_threshold',
    'corrected_resampled_t',
    'edge_names',
    'excess_agreement',
    'fit_lemon_transforms',
    'kendall_agreement',
    'max_stat_permutation',
    'network_atlases',
    'pairwise_accuracy_agreement',
    'permutation_pvalue',
    'random_agreement_baseline',
    'rank_biased_overlap',
    'run_specification',
    'run_specifications',
    'selection_frequency',
    'specification_curve',
    'stability_comparison',
    'stability_vs_performance',
    'summarize',
    'to_ranking',
    'variance_decomposition',
    'vectorize',
    'windowed',
]
