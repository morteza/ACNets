import logging
import sys
from pathlib import Path

import pytest

from acnets.preprocessing import (
  Julia2018BehavioralPreprocessor,
  Julia2018RestingPreprocessor,
  Julia2018TaskPreprocessor,
)

# The rest and task preprocessors convert raw DICOMs, which needs two things
# this checkout does not have: `data/julia2018_raw/`, a pre-BIDS staging tree
# that was never DVC-tracked, and dcm2niix via nipype, which is a system binary
# rather than a declared dependency. The behavioural preprocessor needs neither
# -- it reads `data/julia2018/sourcedata/raw_behavioral/`, which DVC does have --
# so it is not skipped.
needs_raw_dicom = pytest.mark.skipif(
  not Path('data/julia2018_raw').exists(),
  reason='data/julia2018_raw/ is absent and was never DVC-tracked; '
         'conversion also needs dcm2niix. See MERGE_NOTES.md')


@needs_raw_dicom
def test_rest2bids(julia2018_raw_path, bids_path):

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


@needs_raw_dicom
def test_task2bids(julia2018_raw_path, bids_path):
  preprocessor = Julia2018TaskPreprocessor(julia2018_raw_path, bids_path)
  preprocessor.run()

  print('BIDS validation result for the output dataset:', preprocessor.is_valid())
