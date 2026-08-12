from pathlib import Path

import pytest

# `data/julia2018_resting/connectivity_<atlas>.nc` is the input format of the
# superseded loader in acnets.datasets.__julia2018_old, and it was never DVC
# tracked -- it appears in no manifest on the remote, including the last one
# before the root data.dvc was deleted. The current pipeline produces
# `data/Julia2018/timeseries_<atlas>.nc5` instead and nothing outside this test
# and two archived notebooks reads the old format.
#
# Skipped rather than deleted: the loader still works, so if the files turn up
# on the cluster this exercises them.
CONNECTIVITY_DIR = Path('data/julia2018_resting')

pytestmark = pytest.mark.skipif(
    not CONNECTIVITY_DIR.exists(),
    reason=f'{CONNECTIVITY_DIR} is absent and was never DVC-tracked; '
           'see MERGE_NOTES.md')


def test_load_julia2018_connectivity(connectivity_parcellations, connectivity_measures):
  from acnets.datasets.__julia2018_old import load_julia2018_connectivity

  for p in connectivity_parcellations:
    for m in connectivity_measures:
      load_julia2018_connectivity(parcellation=p, kind=m)
