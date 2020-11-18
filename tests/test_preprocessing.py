# to test the Julia2018BehavioralPreprocessor
import logging

import sys


# from .. import codes
from python.preprocessing import Julia2018BehavioralPreprocessor
from python.preprocessing import Julia2018RestingPreprocessor


def test_rest2bids(julia2018_raw_path, julia2018_raw_beh_path, bids_path):

  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

  preprocessor = Julia2018RestingPreprocessor(
      julia2018_raw_path,
      bids_path,
      overwrite=True)
  preprocessor.run()

  preprocessor = Julia2018BehavioralPreprocessor(julia2018_raw_beh_path, bids_path)
  preprocessor.run()

  print('BIDS validation result for the output dataset:',
        preprocessor.is_valid())


def test_task2bids():
  assert True
