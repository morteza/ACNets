"""Declarative enumeration of the analysis multiverse.

The registry is the paper's central scientific object: it states exactly which
analysis paths were considered. It is serialisable so it can ship as a
supplementary file, which is what makes the multiverse claim auditable rather
than merely asserted.

Before this module the grid lived in a notebook cell with most dimensions
commented out (notebooks/ml/2 MultiScale Classifier.ipynb), which is not a
defensible place for it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .specification import (
    ALL_ATLASES,
    ALL_STATISTICS,
    NETWORK_ATLASES,
    Specification,
)

DEFAULT_AGGREGATIONS = ('region', 'network', 'random_network')
# These defaults ARE the declared multiverse of the manuscript, which states its
# specification count as computed from `SpecificationRegistry().summary()` rather
# than asserted. They were briefly narrower than the paper's description -- two
# classifiers and no pretraining axis -- so the bare call returned 280 against a
# manuscript claiming 495. A count a reader is invited to verify has to verify.
DEFAULT_CLASSIFIERS = ('svm', 'xgboost', 'mlp')
DEFAULT_PRETRAININGS = ('none', 'lemon')


@dataclass
class SpecificationRegistry:
    """The set of analysis paths under consideration.

    Invalid combinations are dropped rather than silently run: network-level
    aggregation requires an atlas that defines a region-to-network mapping, so
    region-only atlases (DiFuMo, AAL, Harvard-Oxford) contribute region-level
    specifications only. This is why the grid is not a plain Cartesian product.
    """

    atlases: Sequence[str] = field(default_factory=lambda: tuple(sorted(ALL_ATLASES)))
    statistics: Sequence[str] = field(default_factory=lambda: ALL_STATISTICS)
    aggregations: Sequence[str] = field(default_factory=lambda: DEFAULT_AGGREGATIONS)
    classifiers: Sequence[str] = field(default_factory=lambda: DEFAULT_CLASSIFIERS)
    pretrainings: Sequence[str] = field(default_factory=lambda: DEFAULT_PRETRAININGS)

    def __iter__(self) -> Iterator[Specification]:
        for atlas in self.atlases:
            for statistic in self.statistics:
                for aggregation in self.aggregations:
                    for classifier in self.classifiers:
                        for pretraining in self.pretrainings:
                            spec = Specification(atlas, statistic, aggregation,
                                                 classifier, pretraining)
                            ok, _ = spec.is_valid()
                            if ok:
                                yield spec

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def specifications(self) -> list[Specification]:
        return list(self)

    @property
    def substantive(self) -> list[Specification]:
        """Specifications excluding the random-network negative controls."""
        return [s for s in self if not s.is_control]

    @property
    def controls(self) -> list[Specification]:
        return [s for s in self if s.is_control]

    @property
    def balanced(self) -> list[Specification]:
        """The fully crossed subgrid, for analyses that require orthogonality.

        The full grid is unbalanced by construction: region-only atlases (DiFuMo,
        AAL, Harvard-Oxford) can never appear at network scale, so atlas and
        aggregation are structurally confounded and their contributions cannot be
        separated. Knowing the atlas partly determines which aggregations were
        even available.

        Restricting to the five network-capable atlases restores orthogonality
        between atlas, statistic, aggregation and classifier.

        It does *not* balance `pretraining`, which exists at network scale only
        and, within network scale, only where it is not a no-op
        (`Specification.is_valid`). Any decomposition that includes pretraining
        needs a further restriction -- see the two subgrids in
        `scripts/run_agreement.py` -- so this property is not a general-purpose
        "balanced design", only a balanced one for the original four dimensions.
        Use the full grid for the specification curve and the family-wise null,
        where imbalance is harmless because no dimension is being attributed
        variance.
        """
        return [s for s in self if s.atlas in NETWORK_ATLASES]

    def is_balanced(self) -> bool:
        """Whether every atlas realises every aggregation in this registry."""
        by_atlas: dict[str, set[str]] = {}
        for spec in self:
            by_atlas.setdefault(spec.atlas, set()).add(spec.aggregation)
        if not by_atlas:
            return True
        sizes = {frozenset(v) for v in by_atlas.values()}
        return len(sizes) == 1

    def primary(self,
                atlas: str = 'dosenbach2010',
                statistic: str = 'partial correlation',
                aggregation: str = 'network',
                classifier: str = 'svm') -> Specification:
        """The pre-specified primary analysis.

        Defaults encode the two external justifications given in the paper:
        the Dosenbach parcellation was derived from cognitive control tasks, and
        Liu et al. (2025) found precision-family statistics best on independent
        criteria. Fixing this in code, rather than choosing it after seeing the
        results, is what licenses an uncorrected p-value for this one path.
        """
        spec = Specification(atlas, statistic, aggregation, classifier)
        ok, reason = spec.is_valid()
        if not ok:
            raise ValueError(f'invalid primary specification: {reason}')
        return spec

    def filter(self, **predicates) -> list[Specification]:
        """Select specifications by exact field match.

        >>> registry.filter(aggregation='network', classifier='svm')
        """
        out = []
        for spec in self:
            if all(getattr(spec, k) == v for k, v in predicates.items()):
                out.append(spec)
        return out

    def dimensions(self) -> dict[str, list[str]]:
        """Levels actually realised per dimension, for the variance decomposition."""
        specs = self.specifications
        return {
            dim: sorted({getattr(s, dim) for s in specs})
            for dim in ('atlas', 'statistic', 'aggregation', 'classifier',
                        'pretraining')
        }

    def summary(self) -> dict:
        specs = self.specifications
        by_aggregation: dict[str, int] = {}
        for spec in specs:
            by_aggregation[spec.aggregation] = by_aggregation.get(spec.aggregation, 0) + 1
        return {
            'n_specifications': len(specs),
            'n_substantive': len(self.substantive),
            'n_controls': len(self.controls),
            'by_aggregation': by_aggregation,
            'dimensions': {k: len(v) for k, v in self.dimensions().items()},
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = {
            'summary': self.summary(),
            'axes': {
                'atlases': list(self.atlases),
                'statistics': list(self.statistics),
                'aggregations': list(self.aggregations),
                'classifiers': list(self.classifiers),
                'pretrainings': list(self.pretrainings),
            },
            'specifications': [s.to_dict() for s in self],
        }
        text = json.dumps(payload, indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, path: str | Path) -> SpecificationRegistry:
        payload = json.loads(Path(path).read_text())
        axes = payload['axes']
        return cls(
            atlases=tuple(axes['atlases']),
            statistics=tuple(axes['statistics']),
            aggregations=tuple(axes['aggregations']),
            classifiers=tuple(axes['classifiers']),
            # absent in files written before the axis existed
            pretrainings=tuple(axes.get('pretrainings', ('none',))),
        )

    @classmethod
    def with_external(cls, names: Sequence[str], **kwargs) -> SpecificationRegistry:
        """Add statistics computed outside this codebase and read from disk.

        Kept as an extension point rather than tied to a specific library. Note
        that every statistic added here widens the family-wise correction, so
        add only statistics that differ in their assumptions from the seven
        built in. See the note in pyproject.toml on the pyspi evaluation.
        """
        statistics = tuple(kwargs.pop('statistics', ALL_STATISTICS))
        statistics = statistics + tuple(f'ext:{n}' for n in names)
        return cls(statistics=statistics, **kwargs)


def network_atlases() -> list[str]:
    return sorted(NETWORK_ATLASES)


# The transfer model specified in docs/paper/prereg_primary.md, pinned
# in code so the declaration and the implementation cannot drift apart. This is
# not a Specification: it lives in the timeseries feature space, outside the
# connectivity-grid vocabulary, which is part of what makes it independent of
# every result seen before the declaration was frozen. Frozen 2026-08-07;
# changing any value after that date is a reportable deviation, not an edit.
PRIMARY_CONFIRMATORY = {
    'name': 'lemon-pretrained latent-connectivity classifier',
    'declaration': 'docs/paper/prereg_primary.md',
    'checkpoint': 'checkpoints/lemon_timepoint_e96664/last.pt',
    'checkpoint_id': 'e96664',
    'encoder': {'arch': 'TimepointAE', 'n_channels': 6, 'latent': 4, 'hidden': 16},
    'features': 'latent temporal means + upper-triangle latent covariance (14)',
    'data': 'Julia2018/timeseries_dosenbach2010.nc5, network scale, per-channel z-scored',
    'cv': {'fn': 'fixed_splits', 'n_subjects': 32, 'n_splits': 4,
           'n_repeats': 25, 'random_state': 0},
    'finetune': {'head_only_epochs': 30, 'head_only_lr': 1e-2,
                 'joint_epochs': 30, 'joint_lr': 1e-3,
                 'batch_size': 8, 'seed_base': 1000, 'device': 'cpu'},
    'statistic': 'mean test accuracy over 100 folds',
    'test': {'n_permutations': 1000, 'sided': 'greater', 'alpha': 0.05},
}


def primary_confirmatory() -> dict:
    """The frozen primary, as data. See PRIMARY_CONFIRMATORY."""
    return dict(PRIMARY_CONFIRMATORY)
