"""Execute specifications and record what each one decided.

The runner is deliberately thin. It turns a `Specification` into one immutable
`SpecificationResult` holding per-fold accuracies and a mean attribution vector,
and it does nothing else; every figure is a projection over a collection of
those records, computed by `agreement`, `variance` or `inference`.

Two properties are load-bearing and are the reason this is a module rather than
a notebook cell.

**Nothing is fitted outside the fold.** Connectivity is computed inside the
cross-validation loop, not once up front. For the per-subject statistics this
makes no difference, and it is tempting to hoist it for speed. It is wrong for
`tangent`, which projects each subject onto a group mean covariance estimated in
`fit`: hoisting it lets every test subject contribute to the reference point it
is later measured against. The cost of doing it correctly is a few seconds per
specification.

**Asymmetric statistics keep their lower triangle.** `chatterjee` and
`transfer_entropy` were included precisely because they are directed, and
vectorising them by upper triangle throws away the half of the matrix that
motivated including them. `Specification.is_asymmetric` drives the choice, and
`Specification.n_features` states what the feature count should be, so a silent
symmetrisation shows up as a shape mismatch rather than as a slightly worse
number.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..connectome import ExtraConnectivityMeasure
from ..explain.haufe import average_patterns, haufe_transform
from .specification import Specification

if TYPE_CHECKING:  # transfer.py imports vectorize from here; avoid the cycle
    from .transfer import LemonTransforms

# Separators used to build feature names. The direction of the arrow is the
# only thing distinguishing a directed edge from an undirected one once the
# matrix has been flattened, so it is worth keeping visible in the name.
UNDIRECTED = ' \N{left right arrow} '
DIRECTED = ' \N{rightwards arrow} '


@dataclass(frozen=True)
class SpecificationResult:
    """What one analysis path concluded, as data rather than as a figure."""

    spec: Specification
    fold_accuracies: np.ndarray
    attribution: dict[str, float]
    feature_names: tuple[str, ...]
    n_subjects: int
    n_features: int
    failed: str | None = None

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies))

    def summary(self) -> dict:
        return {
            'spec_id': self.spec.spec_id,
            'label': self.spec.label,
            'atlas': self.spec.atlas,
            'statistic': self.spec.statistic,
            'aggregation': self.spec.aggregation,
            'classifier': self.spec.classifier,
            'is_control': self.spec.is_control,
            'accuracy': self.accuracy,
            'n_folds': int(np.size(self.fold_accuracies)),
            'n_subjects': self.n_subjects,
            'n_features': self.n_features,
        }


def edge_names(nodes: Sequence[str], asymmetric: bool) -> list[str]:
    """Names for the vectorised connectivity matrix, in vectorisation order.

    Symmetric matrices keep the upper triangle including the diagonal, matching
    `Specification.n_features`; asymmetric ones keep every ordered pair.
    """
    nodes = list(nodes)
    if asymmetric:
        return [f'{a}{DIRECTED}{b}' for a in nodes for b in nodes]
    return [f'{nodes[i]}{UNDIRECTED}{nodes[j]}'
            for i in range(len(nodes)) for j in range(i, len(nodes))]


def collapse_to_undirected(attribution: Mapping[str, float]) -> dict[str, float]:
    """Map a directed attribution onto the undirected edge vocabulary.

    Agreement is only defined over a shared vocabulary, and a directed statistic
    does not share one with a symmetric statistic: `CON -> SMN` and
    `CON <-> SMN` are different strings, so rank-biased overlap between a
    chatterjee specification and a correlation specification is exactly zero no
    matter how well they agree. Reporting that zero as disagreement would be a
    naming artefact presented as a finding.

    Each unordered pair therefore keeps whichever direction carries the larger
    magnitude, and keeps it signed:

        A(CON <-> SMN) = A(CON -> SMN) if |A(CON -> SMN)| >= |A(SMN -> CON)|
                         else A(SMN -> CON)

    Maximum rather than mean because attribution is read by magnitude: an edge
    that matters strongly in one direction and not at all in the other matters,
    and averaging would halve it. The collapse is lossy by construction -- it
    discards exactly the directionality that motivated including these
    statistics -- so it belongs to the cross-statistic *comparison* only. The
    directed attribution stays intact in `SpecificationResult.attribution`, and
    anything reported about direction must read that instead.
    """
    collapsed: dict[str, float] = {}
    for name, value in attribution.items():
        if DIRECTED in name:
            source, target = name.split(DIRECTED)
            key = UNDIRECTED.join(sorted((source, target)))
        else:
            key = name
        if key not in collapsed or abs(value) > abs(collapsed[key]):
            collapsed[key] = float(value)
    return collapsed


def vectorize(matrices: np.ndarray, asymmetric: bool) -> np.ndarray:
    """Flatten (n_subjects, n_nodes, n_nodes) into (n_subjects, n_features)."""
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f'expected square matrices, got {matrices.shape}')
    if asymmetric:
        return matrices.reshape(matrices.shape[0], -1)
    rows, cols = np.triu_indices(matrices.shape[1])
    return matrices[:, rows, cols]


def _make_classifier(name: str, random_state: int):
    if name == 'svm':
        return LinearSVC(C=1.0, dual='auto', max_iter=20000,
                         random_state=random_state)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=1,
            eval_metric='logloss')
    if name == 'mlp':
        # xgboost first, always: importing torch ahead of it makes a later
        # XGBoost fit segfault in the same process (acnets._openmp)
        from .._openmp import preload
        preload()
        from ..deep.torch_classifier import TorchMLPClassifier
        return TorchMLPClassifier(random_state=random_state)
    raise ValueError(f'unknown classifier: {name!r}')


def _fold_attribution(name: str,
                      model,
                      x_train: np.ndarray,
                      feature_names: Sequence[str]) -> np.ndarray:
    """Per-fold attribution vector, in the units the classifier admits.

    A linear SVM gets the Haufe forward pattern rather than its own weights,
    because decoder weights depend on the noise structure and cannot be read as
    claims about the brain (Haufe et al. 2014). A tree ensemble has no weights
    to transform, so it gets mean absolute SHAP values, which is the closest
    available analogue: both answer "how strongly does this feature carry the
    discriminative signal", and both are then compared only by rank.
    """
    if name == 'svm':
        return haufe_transform(x_train, model.coef_.ravel())

    if name == 'mlp':
        # closed-form |gradient x input|, comparable in kind to mean |SHAP|;
        # see TorchMLPClassifier.attribution
        return model.attribution(x_train)

    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_train)
    values = np.asarray(values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes) in newer shap
        values = values[..., -1]
    return np.abs(values).mean(axis=0)


def _fold_features(spec: Specification,
                   timeseries: np.ndarray,
                   train_idx: np.ndarray,
                   test_idx: np.ndarray,
                   lemon: LemonTransforms | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled feature matrices for one split.

    Kept free of labels on purpose: nothing in here may receive `y`, which is
    both the leakage argument and what makes the cache below valid across label
    permutations. The one fit that matters is the tangent reference, taken from
    the training fold ('none') or from LEMON ('lemon'); test data is only ever
    transformed.
    """
    if spec.pretraining == 'lemon':
        if lemon is None:
            raise RuntimeError(f'{spec.label} needs LemonTransforms')
        measure = lemon.measure(spec.statistic)  # fitted on LEMON, frozen here
    else:
        measure = ExtraConnectivityMeasure(kind=spec.statistic, vectorize=False)
        measure.fit(timeseries[train_idx])

    x_train = vectorize(measure.transform(timeseries[train_idx]), spec.is_asymmetric)
    x_test = vectorize(measure.transform(timeseries[test_idx]), spec.is_asymmetric)
    return x_train, x_test


def _feature_cache_key(spec: Specification,
                       data_token: str,
                       train_idx: np.ndarray,
                       test_idx: np.ndarray) -> str:
    """Content key for one split's features.

    Deliberately excludes the classifier: specifications differing only there
    share features, which is most of what the cache buys. Includes the data
    token (a hash of the timeseries array) so the same split of different data
    can never collide, and the pretraining flavour, since 'lemon' features
    differ from 'none' features for the reference-based statistics.
    """
    digest = hashlib.sha256()
    for part in (spec.statistic, str(spec.is_asymmetric), spec.pretraining,
                 data_token):
        digest.update(part.encode())
    digest.update(np.ascontiguousarray(train_idx).tobytes())
    digest.update(np.ascontiguousarray(test_idx).tobytes())
    return digest.hexdigest()[:24]


def data_token(timeseries: np.ndarray) -> str:
    """Short content hash of a timeseries array, for the feature cache."""
    return hashlib.sha256(np.ascontiguousarray(timeseries).tobytes()).hexdigest()[:16]


def fixed_splits(n_subjects: int,
                 n_splits: int = 4,
                 n_repeats: int = 25,
                 random_state: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Label-independent cross-validation splits, for permutation testing.

    The default CV inside `run_specification` stratifies on the labels, which
    is the right choice for a single analysis but wrong for a permutation null:
    each permuted label vector would induce its own folds, so (a) the observed
    statistic would be computed on balanced folds while the null is computed on
    whatever the permutation stratifies to -- an asymmetry in the test's own
    favour -- and (b) the feature cache would never hit across permutations,
    since the fold indices are part of its key.

    Deriving the folds from subject indices alone fixes both: observed and
    null runs share one partition that cannot depend on any labelling, the test
    is symmetric conditional on that partition, and features per (statistic,
    fold) are computed exactly once for the entire permutation run. The cost is
    approximate rather than exact class balance within folds, which repetition
    averages out and which observed and null pay identically.
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=random_state)
    return list(cv.split(np.zeros(n_subjects)))


def _cached_fold_features(spec: Specification,
                          timeseries: np.ndarray,
                          train_idx: np.ndarray,
                          test_idx: np.ndarray,
                          lemon: LemonTransforms | None,
                          cache_dir: str | Path | None,
                          token: str | None) -> tuple[np.ndarray, np.ndarray]:
    if cache_dir is None or token is None:
        return _fold_features(spec, timeseries, train_idx, test_idx, lemon)

    key = _feature_cache_key(spec, token, train_idx, test_idx)
    path = Path(cache_dir) / f'{key}.npz'
    if path.exists():
        with np.load(path) as stored:
            return stored['x_train'], stored['x_test']

    x_train, x_test = _fold_features(spec, timeseries, train_idx, test_idx, lemon)
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write: parallel workers may race on the same key, and a torn npz
    # would poison every later run silently. The scratch name must itself end
    # in .npz, because np.savez appends that suffix to anything that lacks it.
    scratch = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(scratch, x_train=x_train, x_test=x_test)
    os.replace(scratch, path)
    return x_train, x_test


def _run_fold(spec: Specification,
              timeseries: np.ndarray,
              labels: np.ndarray,
              names: Sequence[str],
              train_idx: np.ndarray,
              test_idx: np.ndarray,
              fold: int,
              random_state: int,
              compute_attribution: bool = True,
              lemon: LemonTransforms | None = None,
              cache_dir: str | Path | None = None,
              token: str | None = None) -> tuple[float, np.ndarray | None]:
    """One train/test split, start to finish.

    Self-contained by design: it takes indices rather than pre-split arrays and
    builds everything it needs from them, so it can run in a worker process
    without sharing fitted state with any other fold. That is also what makes
    the leakage argument easy to check -- `_fold_features` is the only place
    `train_idx` fits anything, and it never sees a label.
    """
    x_train, x_test = _cached_fold_features(spec, timeseries, train_idx, test_idx,
                                            lemon, cache_dir, token)

    if spec.pretraining == 'lemon':
        scaler = lemon.scaler(spec.statistic)  # fitted on LEMON, frozen
    else:
        scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = _make_classifier(spec.classifier, random_state + fold)
    model.fit(x_train, labels[train_idx])

    accuracy = accuracy_score(labels[test_idx], model.predict(x_test))
    # Attribution is by far the most expensive step for the tree models -- a
    # TreeExplainer pass per fold -- and the permutation null does not use it.
    pattern = (_fold_attribution(spec.classifier, model, x_train, names)
               if compute_attribution else None)
    return float(accuracy), pattern


def run_specification(spec: Specification,
                      timeseries: np.ndarray,
                      labels: np.ndarray,
                      node_names: Sequence[str],
                      n_splits: int = 4,
                      n_repeats: int = 25,
                      random_state: int = 0,
                      n_jobs: int = 1,
                      compute_attribution: bool = True,
                      lemon: LemonTransforms | None = None,
                      cache_dir: str | Path | None = None,
                      splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
                      ) -> SpecificationResult:
    """Fit one specification under repeated stratified cross-validation.

    Parameters
    ----------
    timeseries
        Array of shape (n_subjects, n_timepoints, n_nodes), already aggregated
        to the scale this specification asks for. Aggregation is done outside
        because it is per-subject and therefore cannot leak, and because the
        same aggregated array is reused across every statistic.
    labels
        Binary group labels, shape (n_subjects,).
    n_jobs
        Worker processes for the cross-validation folds. -1 uses every core.
        Folds are the right unit to parallelise over rather than
        specifications: the grid is small and wildly uneven -- an XGBoost
        specification with SHAP attribution costs about four times a linear SVM
        one -- so spreading specifications across cores leaves most of them
        idle waiting for the slowest. The folds within a specification all cost
        the same, which is what keeps every core busy.
    lemon
        Transforms fitted on LEMON, required by (and only by) specifications
        with `pretraining='lemon'`.
    cache_dir
        Directory for the per-fold feature cache. Features are a pure function
        of (statistic, pretraining, data, split) -- never of the labels -- so
        they are identical across label permutations, and the permutation null
        pays only for classifier refits once this is set.
    splits
        Pre-computed (train_idx, test_idx) pairs, e.g. from `fixed_splits`.
        When omitted, repeated stratified k-fold on the labels is used, which
        is right for a single analysis but must NOT be used inside a
        permutation loop -- see `fixed_splits` for why.

    Returns
    -------
    SpecificationResult
        With `failed` set to the reason string, and empty accuracies, if the
        specification could not be run. A specification that fails is recorded
        rather than dropped, because silently shrinking the grid changes the
        family-wise correction.

    Notes
    -----
    Results do not depend on `n_jobs`. Each fold's classifier is seeded from
    `random_state + fold`, and folds are collected in the order joblib was
    handed them, so a parallel run reproduces a serial one exactly.
    """
    ok, reason = spec.is_valid()
    if not ok:
        return SpecificationResult(spec, np.array([]), {}, (), len(labels), 0, reason)
    if spec.pretraining == 'lemon':
        if lemon is None:
            return SpecificationResult(spec, np.array([]), {}, (), len(labels), 0,
                                       'lemon pretraining requested but no '
                                       'LemonTransforms supplied')
        lemon.check_nodes(node_names)

    n_nodes = timeseries.shape[-1]
    names = edge_names(node_names, spec.is_asymmetric)
    expected = spec.n_features(n_nodes)
    if len(names) != expected:
        raise ValueError(
            f'{spec.label}: built {len(names)} feature names but '
            f'Specification.n_features says {expected}')

    if splits is None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=random_state)
        splits = list(cv.split(timeseries, labels))

    token = data_token(timeseries) if cache_dir is not None else None

    jobs = (delayed(_run_fold)(spec, timeseries, labels, names,
                               train_idx, test_idx, fold, random_state,
                               compute_attribution, lemon, cache_dir, token)
            for fold, (train_idx, test_idx) in enumerate(splits))

    if n_jobs == 1:
        folds = [job[0](*job[1], **job[2]) for job in jobs]
    else:
        # inner_max_num_threads=1 stops each worker's BLAS from opening its own
        # thread pool. Without it 12 workers each spawn 12 threads, and the
        # oversubscription costs more than the parallelism gains.
        with parallel_config(backend='loky', inner_max_num_threads=1):
            folds = Parallel(n_jobs=n_jobs)(jobs)

    accuracies = [accuracy for accuracy, _ in folds]
    patterns = [pattern for _, pattern in folds if pattern is not None]

    # strict: one attribution value per feature name is an invariant, not a
    # coincidence. A mismatch means the vectorisation and the naming disagree
    # about the matrix shape, and zipping silently to the shorter of the two
    # would drop features off the end of the ranking without saying so.
    attribution = (dict(zip(names, average_patterns(patterns).tolist(), strict=True))
                   if patterns else {})

    return SpecificationResult(
        spec=spec,
        fold_accuracies=np.asarray(accuracies, dtype=float),
        attribution=attribution,
        feature_names=tuple(names),
        n_subjects=len(labels),
        n_features=len(names),
    )


@dataclass
class RunOutcome:
    """Everything a projection needs, keyed by spec_id."""

    results: dict[str, SpecificationResult] = field(default_factory=dict)

    @property
    def accuracies(self) -> dict[str, float]:
        return {k: r.accuracy for k, r in self.results.items() if r.failed is None}

    @property
    def attributions(self) -> dict[str, dict[str, float]]:
        return {k: r.attribution for k, r in self.results.items() if r.failed is None}

    def labelled_accuracies(self) -> dict[str, float]:
        return {r.spec.label: r.accuracy
                for r in self.results.values() if r.failed is None}

    def labelled_attributions(self) -> dict[str, dict[str, float]]:
        return {r.spec.label: r.attribution
                for r in self.results.values() if r.failed is None}

    def add(self, result: SpecificationResult) -> None:
        self.results[result.spec.spec_id] = result


def run_specifications(specs: Sequence[Specification],
                       timeseries_by_aggregation: Mapping[str, np.ndarray],
                       node_names_by_aggregation: Mapping[str, Sequence[str]],
                       labels: np.ndarray,
                       progress=None,
                       **kwargs) -> RunOutcome:
    """Run a set of specifications that share one atlas.

    `timeseries_by_aggregation` maps each aggregation ('region', 'network',
    'random_network') to its already-aggregated timeseries array, so the
    aggregation is computed once and reused across statistics and classifiers.
    """
    outcome = RunOutcome()
    for spec in specs:
        if progress is not None:
            progress(spec)
        result = run_specification(
            spec,
            timeseries_by_aggregation[spec.aggregation],
            labels,
            node_names_by_aggregation[spec.aggregation],
            **kwargs)
        outcome.add(result)
    return outcome
