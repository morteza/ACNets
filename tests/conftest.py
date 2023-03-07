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


@pytest.fixture
def data_path():
  return Path('data/')


@pytest.fixture
def connectivity_parcellations():
  from src.acnets.datasets import __supported_parcellations
  return __supported_parcellations


@pytest.fixture
def connectivity_measures():
  from src.acnets.datasets import __supported_kinds
  return __supported_kinds
