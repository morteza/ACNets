import numpy as np
import pytest

from .runner import (
    DIRECTED,
    UNDIRECTED,
    collapse_to_undirected,
    edge_names,
    run_specification,
    run_specifications,
    vectorize,
)
from .specification import Specification


@pytest.fixture
def nodes():
    return ['CON', 'FPN', 'SMN']


@pytest.fixture
def toy_data():
    """Three networks, two groups, with a planted difference on CON-SMN."""
    rng = np.random.default_rng(0)
    n_per_group, n_timepoints = 12, 80

    def make(coupling):
        shared = rng.normal(size=(n_per_group, n_timepoints))
        con = shared + rng.normal(scale=0.5, size=(n_per_group, n_timepoints))
        smn = coupling * shared + rng.normal(scale=0.5, size=(n_per_group, n_timepoints))
        fpn = rng.normal(size=(n_per_group, n_timepoints))
        return np.stack([con, fpn, smn], axis=-1)

    x = np.concatenate([make(1.5), make(0.0)])
    y = np.array([0] * n_per_group + [1] * n_per_group)
    return x, y


def test_edge_names_symmetric_matches_n_features(nodes):
    spec = Specification('dosenbach2010', 'correlation', 'network')
    names = edge_names(nodes, asymmetric=False)
    assert len(names) == spec.n_features(len(nodes)) == 6
    assert names[0] == f'CON{UNDIRECTED}CON'
    assert names[1] == f'CON{UNDIRECTED}FPN'


def test_edge_names_asymmetric_keeps_both_directions(nodes):
    spec = Specification('dosenbach2010', 'chatterjee', 'network')
    names = edge_names(nodes, asymmetric=True)
    assert len(names) == spec.n_features(len(nodes)) == 9
    assert f'CON{DIRECTED}SMN' in names
    assert f'SMN{DIRECTED}CON' in names


def test_vectorize_symmetric_takes_upper_triangle_with_diagonal():
    m = np.arange(9, dtype=float).reshape(1, 3, 3)
    m = m + m.transpose(0, 2, 1)
    out = vectorize(m, asymmetric=False)
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out[0], [0., 4., 8., 8., 12., 16.])


def test_vectorize_asymmetric_keeps_every_ordered_pair():
    m = np.arange(9, dtype=float).reshape(1, 3, 3)
    out = vectorize(m, asymmetric=True)
    assert out.shape == (1, 9)
    np.testing.assert_allclose(out[0], np.arange(9))


def test_vectorize_rejects_non_square():
    with pytest.raises(ValueError, match='square'):
        vectorize(np.zeros((2, 3, 4)), asymmetric=False)


def test_collapse_keeps_the_stronger_direction_signed():
    directed = {
        f'CON{DIRECTED}SMN': 0.2,
        f'SMN{DIRECTED}CON': -0.9,
        f'CON{DIRECTED}CON': 0.5,
    }
    collapsed = collapse_to_undirected(directed)
    assert collapsed[f'CON{UNDIRECTED}SMN'] == -0.9
    assert collapsed[f'CON{UNDIRECTED}CON'] == 0.5


def test_collapse_leaves_undirected_attributions_untouched():
    undirected = {f'CON{UNDIRECTED}SMN': 0.3, f'FPN{UNDIRECTED}SMN': -0.1}
    assert collapse_to_undirected(undirected) == undirected


def test_collapse_makes_directed_and_undirected_share_a_vocabulary(nodes):
    symmetric = dict.fromkeys(edge_names(nodes, asymmetric=False), 1.0)
    directed = dict.fromkeys(edge_names(nodes, asymmetric=True), 1.0)
    assert set(collapse_to_undirected(directed)) == set(symmetric)


def test_agreement_between_directed_and_undirected_is_zero_without_collapse(nodes):
    """The artefact collapse_to_undirected exists to remove."""
    from .agreement import agreement_matrix

    symmetric = {n: float(i) for i, n in enumerate(edge_names(nodes, False))}
    directed = {n: float(i) for i, n in enumerate(edge_names(nodes, True))}

    raw = agreement_matrix({'sym': symmetric, 'dir': directed})
    assert raw.mean == 0.0

    collapsed = agreement_matrix({'sym': symmetric,
                                  'dir': collapse_to_undirected(directed)})
    assert collapsed.mean > 0.0


def test_run_specification_recovers_planted_signal(toy_data, nodes):
    x, y = toy_data
    spec = Specification('dosenbach2010', 'correlation', 'network', 'svm')
    result = run_specification(spec, x, y, nodes, n_splits=3, n_repeats=2)

    assert result.failed is None
    assert result.accuracy > 0.7
    assert set(result.attribution) == set(result.feature_names)
    assert result.n_features == spec.n_features(len(nodes))

    top = max(result.attribution, key=lambda k: abs(result.attribution[k]))
    assert top == f'CON{UNDIRECTED}SMN'


def test_run_specification_records_invalid_rather_than_raising(toy_data, nodes):
    x, y = toy_data
    spec = Specification('aal', 'correlation', 'network')  # aal defines no networks
    result = run_specification(spec, x, y, nodes)

    assert result.failed is not None
    assert result.fold_accuracies.size == 0


def test_run_specification_asymmetric_statistic_keeps_full_matrix(toy_data, nodes):
    x, y = toy_data
    spec = Specification('dosenbach2010', 'chatterjee', 'network', 'svm')
    result = run_specification(spec, x, y, nodes, n_splits=3, n_repeats=1)

    assert result.n_features == 9
    assert result.n_features == spec.n_features(len(nodes))


def test_tangent_reference_is_not_refitted_on_the_test_fold(toy_data, nodes):
    """A tangent fit on train must not be replaced by one fit on test.

    The failure this guards against is silent: re-fitting on the test fold
    still produces a plausible accuracy, so nothing looks wrong.
    """
    from ..connectome import ExtraConnectivityMeasure

    x, _ = toy_data
    measure = ExtraConnectivityMeasure(kind='tangent', vectorize=False)
    measure.fit(x[:12])

    from_train_reference = measure.transform(x[12:])
    refitted = measure.fit(x[12:]).transform(x[12:])

    assert not np.allclose(from_train_reference, refitted)


def test_parallel_folds_reproduce_the_serial_result(toy_data, nodes):
    """n_jobs must be a performance knob, never a source of different numbers."""
    x, y = toy_data
    spec = Specification('dosenbach2010', 'correlation', 'network', 'svm')
    kwargs = dict(n_splits=4, n_repeats=3, random_state=7)

    serial = run_specification(spec, x, y, nodes, n_jobs=1, **kwargs)
    parallel = run_specification(spec, x, y, nodes, n_jobs=2, **kwargs)

    np.testing.assert_allclose(serial.fold_accuracies, parallel.fold_accuracies)
    assert serial.attribution.keys() == parallel.attribution.keys()
    for name in serial.attribution:
        np.testing.assert_allclose(serial.attribution[name],
                                   parallel.attribution[name])


def test_run_specifications_collects_by_spec_id(toy_data, nodes):
    x, y = toy_data
    specs = [Specification('dosenbach2010', s, 'network', 'svm')
             for s in ('correlation', 'covariance')]
    outcome = run_specifications(
        specs,
        {'network': x},
        {'network': nodes},
        y,
        n_splits=3, n_repeats=1)

    assert set(outcome.results) == {s.spec_id for s in specs}
    assert set(outcome.accuracies) == {s.spec_id for s in specs}
    assert len(outcome.labelled_attributions()) == 2
