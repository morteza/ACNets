"""A single point in the analysis multiverse.

A Specification is an immutable, content-addressed description of one analysis
path: which parcellation defines the nodes, which statistic defines the edges,
at what scale connectivity is aggregated, and which classifier is fitted.

Content addressing matters: spec_id is a hash of the content, so the runner can
be interrupted and resumed, and a results store can be keyed by specification
without a separate registry lookup.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

Aggregation = Literal['region', 'network', 'random_network']
# 'mlp' is a small torch network behind the same sklearn surface
# (acnets.deep.torch_classifier); it needs the dl extra installed.
ClassifierName = Literal['svm', 'xgboost', 'mlp']

# Where fold-independent state comes from. 'none' fits everything inside the
# training fold, as before. 'lemon' fits the connectivity reference (tangent's
# projection point) and the feature scaler on MPI-LEMON, an external resting
# dataset with no subjects in common with Julia2018 -- transfer that cannot
# leak, because the labels it never sees are the labels being predicted.
Pretraining = Literal['none', 'lemon']

# Atlases that define a region-to-network mapping. Network-level and
# random-network aggregation are only meaningful for these; the others expose
# regions only. Keep in sync with src/acnets/parcellations/.
NETWORK_ATLASES = frozenset({
    'dosenbach2010',
    'gordon2014',
    'seitzman2018',
    'friedman2020',
    'yeo2011',
})

REGION_ONLY_ATLASES = frozenset({
    'difumo_64',
    'difumo_128',
    'difumo_256',
    'aal',
    'cort-maxprob-thr25-2mm',
})

ALL_ATLASES = NETWORK_ATLASES | REGION_ONLY_ATLASES

# Seven statistics chosen to span distinct assumption classes rather than to
# maximise count:
#
#   correlation, covariance    linear, symmetric
#   partial correlation        conditional, symmetric
#   precision                  conditional inverse, sparse
#   tangent                    Riemannian projection of covariance
#   chatterjee                 rank-based, detects non-monotonic dependence
#   transfer_entropy           directed, information theoretic
#
# Breadth for its own sake is counterproductive here: the family-wise
# max-statistic correction in acnets.multiverse.inference punishes redundancy, so
# adding many correlated variants of covariance raises the bar the observed
# effect must clear without adding independent information. See the note in
# pyproject.toml on why the pyspi library was evaluated and dropped.
#
# All seven are implemented in acnets.connectome.ExtraConnectivityMeasure.
NILEARN_STATISTICS = (
    'correlation',
    'partial correlation',
    'covariance',
    'precision',
    'tangent',
)
EXTRA_STATISTICS = ('chatterjee', 'transfer_entropy')
ALL_STATISTICS = NILEARN_STATISTICS + EXTRA_STATISTICS

# Statistics that produce a directed matrix. Vectorising by upper triangle is
# lossy for these, which is the whole point of including them.
ASYMMETRIC_STATISTICS = frozenset({'chatterjee', 'transfer_entropy'})

# Classifiers whose predictions change when the features are affinely rescaled.
# A tree ensemble splits on order statistics, so any monotone per-feature
# rescaling leaves it bit-identical -- which is what makes some pretraining
# combinations degenerate; see Specification.is_valid.
SCALE_SENSITIVE_CLASSIFIERS = frozenset({'svm', 'mlp'})

# The one statistic whose `fit` carries state, so the only one for which LEMON
# pretraining changes the connectivity matrix itself rather than only its
# scaling. Tangent projection needs a reference covariance; estimating it from
# 220 external subjects instead of 24 training subjects is the substantive part
# of the transfer.
STATEFUL_STATISTICS = frozenset({'tangent'})


@dataclass(frozen=True)
class Specification:
    """One analysis path through the multiverse.

    Parameters
    ----------
    atlas
        Parcellation name, as accepted by acnets.pipeline.Parcellation.
    statistic
        Connectivity statistic. Normally one of ALL_STATISTICS. A statistic
        computed by an external tool and read from disk may be named with an
        'ext:' prefix, which bypasses the vocabulary check.
    aggregation
        'region' keeps region-by-region connectivity. 'network' averages
        timeseries within networks before computing connectivity.
        'random_network' does the same but with regions shuffled across
        size-matched random networks, and is the dimensionality-matched
        negative control.
    classifier
        Classifier family.
    """

    atlas: str
    statistic: str
    aggregation: Aggregation
    classifier: ClassifierName = 'svm'
    # NOTE adding this field changed every spec_id (the id is a content hash).
    # Stored results keyed by old ids are regenerable projections, not data,
    # so nothing is migrated; rerun the drivers.
    pretraining: Pretraining = 'none'

    def __post_init__(self):
        if self.aggregation not in ('region', 'network', 'random_network'):
            raise ValueError(f'unknown aggregation: {self.aggregation!r}')
        if self.pretraining not in ('none', 'lemon'):
            raise ValueError(f'unknown pretraining: {self.pretraining!r}')

    @property
    def spec_id(self) -> str:
        """Stable 12-character content hash. Order-independent and portable."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def is_control(self) -> bool:
        """True for the random-network negative control."""
        return self.aggregation == 'random_network'

    @property
    def is_external(self) -> bool:
        """Statistic computed outside this codebase and read from disk."""
        return self.statistic.startswith('ext:')

    @property
    def is_asymmetric(self) -> bool:
        """Whether the statistic produces a directed (non-symmetric) matrix.

        Matters because vectorising a connectivity matrix by taking its upper
        triangle is only lossless when the matrix is symmetric. Transfer entropy
        is directed by construction, TE(i -> j) != TE(j -> i), and Chatterjee's
        xi is likewise asymmetric in its arguments. Discarding the lower triangle
        for these two throws away exactly the information that motivated
        including them.

        Note that ExtraConnectivityMeasure already refuses `vectorize=True` for
        transfer entropy for this reason, but the separate vectorisation step in
        MultiScaleClassifier applies np.triu_indices unconditionally. Callers
        must branch on this property; see the note in section 6.4 of the
        manuscript.
        """
        return self.statistic in ASYMMETRIC_STATISTICS

    def n_features(self, n_nodes: int) -> int:
        """Number of classifier input features for a given node count.

        Reported per specification so dimensionality is never silently
        confounded with statistic in the comparisons of section 3.2: an
        asymmetric statistic over n nodes yields n*n features, against
        n*(n+1)/2 for a symmetric one including the diagonal.
        """
        if n_nodes < 1:
            raise ValueError('n_nodes must be positive')
        if self.is_asymmetric:
            return n_nodes * n_nodes
        return n_nodes * (n_nodes + 1) // 2

    def is_valid(self) -> tuple[bool, str]:
        """Whether this specification can actually be run.

        Returns (ok, reason). Network-level aggregation needs an atlas that
        defines networks; tangent projection needs more than one subject per
        fold and is checked at run time, not here.
        """
        if self.atlas not in ALL_ATLASES:
            return False, f'unknown atlas {self.atlas!r}'
        if (self.aggregation in ('network', 'random_network')
                and self.atlas not in NETWORK_ATLASES):
            return False, f'{self.atlas!r} defines no region-to-network mapping'
        if not self.is_external and self.statistic not in ALL_STATISTICS:
            return False, f'unknown statistic {self.statistic!r}'
        if self.pretraining == 'lemon':
            # The LEMON transforms are fitted on real-network aggregation of a
            # network atlas, so they only apply where the feature vocabulary
            # matches. random_network shuffles the vocabulary per run, and
            # region scale is a different feature space entirely (deferred).
            if self.aggregation != 'network':
                return False, 'lemon pretraining is defined at network scale only'
            if self.is_external:
                return False, 'lemon pretraining does not cover external statistics'
            # Degenerate cells. LEMON pretraining changes two things: the
            # connectivity measure (only for a stateful statistic) and the
            # feature scaler (only visible to a scale-sensitive classifier). If
            # neither applies, this specification is the *same computation* as
            # its 'none' twin, and emitting it would put an exact duplicate in
            # the grid -- inflating the reported number of analysis paths,
            # contributing a free 1.0 to mean attribution agreement, and
            # diluting the pretraining dimension in the variance decomposition
            # with cells that are zero by construction.
            if (self.statistic not in STATEFUL_STATISTICS
                    and self.classifier not in SCALE_SENSITIVE_CLASSIFIERS):
                return False, (f'lemon pretraining is a no-op for '
                               f'{self.statistic!r} with {self.classifier!r}: '
                               f'stateless measure, scale-invariant classifier')
        return True, ''

    @property
    def label(self) -> str:
        """Short human-readable label for figures."""
        stat = self.statistic.replace('ext:', '').replace('partial correlation', 'pcorr')
        base = f'{self.atlas}/{stat}/{self.aggregation}/{self.classifier}'
        return base if self.pretraining == 'none' else f'{base}+{self.pretraining}'

    def to_dict(self) -> dict:
        return {'spec_id': self.spec_id, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict) -> Specification:
        fields = {k: v for k, v in payload.items() if k != 'spec_id'}
        return cls(**fields)
