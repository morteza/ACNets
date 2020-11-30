import pytest

from pathlib import Path


@pytest.fixture
def julia2018_raw_path():
  return Path('data/julia2018_raw')


@pytest.fixture
def julia2018_raw_beh_path():
  return Path('data/julia2018/sourcedata/raw_behavioral')


@pytest.fixture
def bids_path():
  return Path('data/julia2018')
