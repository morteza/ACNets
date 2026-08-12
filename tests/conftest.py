from pathlib import Path

import pytest


@pytest.fixture
def julia2018_raw_path():
  return Path('data/julia2018_raw')


@pytest.fixture
def julia2018_raw_beh_path():
  return Path('data/julia2018/sourcedata/raw_behavioral')


@pytest.fixture
def bids_path(tmp_path):
  """Where the BIDS-building tests write, which must not be the real dataset.

  This used to be `data/julia2018` itself, so `test_beh2bids` rewrote the
  tracked `sub-*/ses-*/beh/*.tsv` files on every run. The values it produced
  were right, but pandas now formats floats with fewer digits, so every file
  came back byte-different from what DVC has and the dataset looked modified
  after merely running the tests.

  The preprocessors expect to write into an initialised BIDS root -- pybids
  refuses a directory with no dataset_description.json -- so the fixture lays
  one down first.
  """
  from acnets.preprocessing.utils import init_bids

  root = tmp_path / 'julia2018'
  init_bids(root)
  return root


@pytest.fixture
def data_path():
  return Path('data/')


# Both lists live in acnets.datasets.__julia2018_old, the module the connectivity
# loader under test comes from; acnets.datasets itself never re-exported them.
@pytest.fixture
def connectivity_parcellations():
  from acnets.datasets.__julia2018_old import __supported_parcellations
  return __supported_parcellations


@pytest.fixture
def connectivity_measures():
  from acnets.datasets.__julia2018_old import __supported_kinds
  return __supported_kinds
