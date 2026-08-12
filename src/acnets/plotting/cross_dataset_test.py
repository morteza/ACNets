"""Tests for the external-validation figure.

The figure's job is to show six accuracies against six nulls. Its one real
failure mode is drawing an observed point with no null behind it -- the reader
would see a number at 0.51 and have nothing to judge it against, which is
exactly the misreading the figure exists to prevent. Older stores predate the
`null_median`/`null_95` keys, so that case is tested explicitly.
"""

import json

import matplotlib
import pytest

matplotlib.use('Agg')

from acnets.plotting.cross_dataset import cross_dataset_figure


def _payload(atlas, profile=0.87, **overrides):
    results = {
        'within/gong2015': 0.574, 'within/julia2018': 0.438,
        'transfer/gong2015->julia2018': 0.500,
        'p/gong2015->julia2018': 0.775,
        'null_median/gong2015->julia2018': 0.500,
        'null_95/gong2015->julia2018': 0.594,
        'transfer/julia2018->gong2015': 0.511,
        'p/julia2018->gong2015': 0.505,
        'null_median/julia2018->gong2015': 0.511,
        'null_95/julia2018->gong2015': 0.532,
    }
    results.update(overrides)
    return {'atlas': atlas, 'profile_correlation': profile,
            'n': {'gong2015': 47, 'julia2018': 32},
            'n_permutations': 1000, 'results': results}


def _write(directory, *payloads):
    for payload in payloads:
        (directory / f'cross_dataset_{payload["atlas"]}.json').write_text(
            json.dumps(payload))


class TestFigure:

    def test_draws_two_rows_per_atlas(self, tmp_path):
        _write(tmp_path, _payload('aal'), _payload('harvard-oxford', 0.94))
        figure = cross_dataset_figure(tmp_path)
        left = figure.axes[0]
        assert len(left.get_yticks()) == 4

    def test_orders_atlases_the_same_way_in_both_panels(self, tmp_path):
        _write(tmp_path, _payload('schaefer200', 0.58), _payload('aal'))
        left, right = cross_dataset_figure(tmp_path).axes
        row_atlases = [t.get_text().split('\n')[0]
                       for t in left.get_yticklabels()]
        bar_atlases = [t.get_text() for t in right.get_yticklabels()]
        assert row_atlases[::2] == bar_atlases

    def test_refuses_to_draw_an_accuracy_without_its_null(self, tmp_path):
        payload = _payload('aal')
        del payload['results']['null_median/gong2015->julia2018']
        _write(tmp_path, payload)
        with pytest.raises(KeyError, match='null_median'):
            cross_dataset_figure(tmp_path)

    def test_says_what_to_run_when_the_directory_is_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='run_cross_dataset'):
            cross_dataset_figure(tmp_path)

    def test_x_range_covers_every_null_it_draws(self, tmp_path):
        """A null reaching past the axis would be silently clipped."""
        _write(tmp_path, _payload(
            'aal', **{'transfer/julia2018->gong2015': 0.489}))
        left = cross_dataset_figure(tmp_path).axes[0]
        low, high = left.get_xlim()
        assert low <= 0.489 and high >= 0.594
