# to test the Julia2018BehavioralPreprocessor
import logging

import sys

from pathlib import Path

# from .. import codes
from python.preprocessing import Julia2018BehavioralPreprocessor
from python.preprocessing import Julia2018RestingPreprocessor


def test_bids_conversion():

  orig_dir = Path('data/julia2018_raw')
  raw_beh_dir = Path('data/julia2018/raw_beh')
  bids_dir = Path('data/julia2018/bids')

  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

  preprocessor = Julia2018RestingPreprocessor(
      orig_dir,
      bids_dir,
      overwrite=True)
  preprocessor.run()

  preprocessor = Julia2018BehavioralPreprocessor(raw_beh_dir, bids_dir)
  preprocessor.run()

  print('BIDS validation result for the output dataset:',
        preprocessor.is_valid())
