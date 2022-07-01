import logging

import sys




def test_rest2bids(julia2018_raw_path, bids_path):

  # imports
  from python.acnets.preprocessing import (Julia2018BehavioralPreprocessor,
                                          Julia2018RestingPreprocessor,
                                          Julia2018TaskPreprocessor)

  # logging
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

  preprocessor = Julia2018RestingPreprocessor(
      julia2018_raw_path,
      bids_path,
      overwrite=True)
  preprocessor.run()

  print('BIDS validation result for the output dataset:', preprocessor.is_valid())


def test_beh2bids(julia2018_raw_beh_path, bids_path):
  preprocessor = Julia2018BehavioralPreprocessor(julia2018_raw_beh_path, bids_path)
  preprocessor.run()

  print('BIDS validation result for the output dataset:', preprocessor.is_valid())


def test_task2bids(julia2018_raw_path, bids_path):
  preprocessor = Julia2018TaskPreprocessor(julia2018_raw_path, bids_path)
  preprocessor.run()

  print('BIDS validation result for the output dataset:', preprocessor.is_valid())
