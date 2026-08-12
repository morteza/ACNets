"""Tests for the specification-curve figure, focused on the label parsing.

The dot matrix is only meaningful if column i really is specification i, and
that alignment rests entirely on parsing the label strings correctly. A silent
parse failure would produce a plausible-looking but wrong figure, which is the
worst outcome for a figure whose whole job is to show which choices matter.
"""

import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from acnets.plotting.specification_curve import (
    parse_labels,
    specification_curve_figure,
)


class TestParseLabels:

    def test_parses_a_plain_label(self):
        out = parse_labels(pd.Series(['dosenbach2010/pcorr/network/xgboost']))
        assert out.iloc[0].to_dict() == {
            'atlas': 'dosenbach2010', 'statistic': 'pcorr',
            'aggregation': 'network', 'classifier': 'xgboost',
            'pretraining': 'none'}

    def test_parses_a_pretraining_suffix(self):
        out = parse_labels(pd.Series(['gordon2014/tangent/network/svm+lemon']))
        assert out.iloc[0]['pretraining'] == 'lemon'
        assert out.iloc[0]['classifier'] == 'svm'

    def test_handles_a_statistic_containing_a_space(self):
        out = parse_labels(pd.Series(['aal/partial correlation/region/mlp']))
        assert out.iloc[0]['statistic'] == 'partial correlation'

    def test_parses_the_three_part_structural_label(self):
        """The gong2015 arm has no aggregation axis."""
        out = parse_labels(pd.Series(['aal/gmv/xgboost']))
        assert out.iloc[0]['aggregation'] == 'n/a'
        assert out.iloc[0]['classifier'] == 'xgboost'

    def test_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match='cannot parse'):
            parse_labels(pd.Series(['dosenbach2010/pcorr']))

    def test_preserves_row_order(self):
        labels = pd.Series(['a/b/c/d', 'e/f/g/h', 'i/j/k/l'])
        assert list(parse_labels(labels)['atlas']) == ['a', 'e', 'i']


class TestFigure:

    def _write(self, directory, significant=(False, False, False, False)):
        rows = [
            {'spec_id': 'dosenbach2010/pcorr/network/svm', 'accuracy': 0.52},
            {'spec_id': 'dosenbach2010/tangent/network/svm+lemon', 'accuracy': 0.58},
            {'spec_id': 'dosenbach2010/pcorr/random_network/xgboost', 'accuracy': 0.47},
            {'spec_id': 'dosenbach2010/tangent/network/mlp', 'accuracy': 0.66},
        ]
        frame = pd.DataFrame(rows)
        frame['significant'] = list(significant)
        frame['rank'] = frame['accuracy'].rank().astype(int)
        frame.to_csv(directory / 'specification_curve.csv', index=False)
        # key names must match MaxStatResult.summary() exactly
        (directory / 'permutation.json').write_text(json.dumps(
            {'critical_value_95': 0.64, 'n_permutations': 1000,
             'observed_max': 0.66, 'p_value': 0.04}))

    def test_builds_a_figure_with_two_aligned_panels(self, tmp_path):
        self._write(tmp_path)
        figure = specification_curve_figure(tmp_path)
        top, bottom = figure.axes
        assert top.get_xlim() == bottom.get_xlim()

    def test_dot_matrix_skips_single_level_dimensions(self, tmp_path):
        """Only one atlas is present, so no atlas rows should be drawn."""
        self._write(tmp_path)
        figure = specification_curve_figure(tmp_path)
        _, bottom = figure.axes
        labels = [t.get_text() for t in bottom.get_yticklabels()]
        assert 'dosenbach2010' not in labels
        assert {'network', 'random_network'} <= set(labels)
        assert {'none', 'lemon'} <= set(labels)

    def test_title_reports_how_many_cleared_the_null(self, tmp_path):
        self._write(tmp_path, significant=(False, False, False, True))
        figure = specification_curve_figure(tmp_path)
        assert '1 clearing' in figure.axes[0].get_title(loc='left')

    def test_refuses_to_draw_without_the_critical_value(self, tmp_path):
        """A missing threshold silently produced a figure with no threshold."""
        self._write(tmp_path)
        (tmp_path / 'permutation.json').write_text(json.dumps(
            {'n_permutations': 1000, 'observed_max': 0.66}))
        with pytest.raises(FileNotFoundError, match='critical_value_95'):
            specification_curve_figure(tmp_path)

    def test_finds_a_summary_under_another_name(self, tmp_path):
        """The gong2015 arm writes gong2015.json, not permutation.json."""
        self._write(tmp_path)
        (tmp_path / 'permutation.json').rename(tmp_path / 'gong2015.json')
        figure = specification_curve_figure(tmp_path)
        ys = [line.get_ydata()[0] for line in figure.axes[0].lines]
        assert 0.64 in ys

    def test_draws_the_critical_value_line(self, tmp_path):
        self._write(tmp_path)
        figure = specification_curve_figure(tmp_path)
        ys = [line.get_ydata()[0] for line in figure.axes[0].lines]
        assert 0.64 in ys

    def test_sorts_by_accuracy(self, tmp_path):
        self._write(tmp_path)
        figure = specification_curve_figure(tmp_path)
        # the scatter of non-significant points carries the sorted accuracies
        offsets = figure.axes[0].collections[-1].get_offsets()
        accuracies = [point[1] for point in offsets]
        assert accuracies == sorted(accuracies)
