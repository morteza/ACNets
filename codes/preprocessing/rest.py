import logging

from os import PathLike
from pathlib import Path
import shutil

import json

from dataclasses import dataclass

from bids.layout import BIDSLayout
from bids.layout import BIDSValidator

from .utils import dcm2bids


@dataclass
class Julia2018RestingPreprocessor():
  """Prepares BIDS-compatible resting state data."""

  in_dir: PathLike
  out_dir: PathLike
  session: str = 'rest'
  task_name: str = 'rest'
  overwrite: bool = False

  def __post_init__(self) -> None:

    self.in_dir = Path(self.in_dir)
    self.out_dir = Path(self.out_dir)

    if (not self.overwrite) and self.out_dir.exists():
      raise Exception('Output directory exists. Either set the `overwrite`'
                      ' or use a different output dirctory.')

    self.rest_dir = self.in_dir / 'Resting_State' / 'Resting_State'

    # TODO filter subjects with NEW suffix and append `acq` instead.

    # key: sub; value: folder_name (julia_sub)
    self.subjects = {
        # rename VGP/NVGP => AVGP/NVGP
        s.stem.replace('VGP', 'AVGP')
        if s.stem.startswith('VGP') else s.stem: s.stem
        for s in self.rest_dir.iterdir() if s.is_dir()
    }

  def run(self):
    """Runs bidifier for all subjects."""

    for sub, julia_sub in self.subjects.items():
      self.init_folder_structure(sub)
      self.copy_rest_t1w(sub, julia_sub)
      self.convert_rest_bold(sub, julia_sub)
      self.copy_rest_fmap(sub, julia_sub)

    self.create_task_sidecar()
    self.create_dataset_description()

    logging.info(
        'BIDS-ified Julia2018 Resting State (bold, T1w, fmap)'
    )

  def init_folder_structure(self, sub):
    """Creates initial folder structure for a given subject."""

    ses_dir = self.out_dir / f'sub-{sub}' / f'ses-{self.session}'
    (ses_dir / 'anat').mkdir(parents=True, exist_ok=True)
    (ses_dir / 'func').mkdir(parents=True, exist_ok=True)
    (ses_dir / 'fmap').mkdir(parents=True, exist_ok=True)

  def convert_rest_bold(self, sub, julia_sub):
    """convert BOLD DICOMs into nifti/sidecar, and move them into func/."""

    bids_func = self.out_dir / f'sub-{sub}' / f'ses-{self.session}' / 'func'

    # TODO: use heudiconv to convery DCM files

    # precompiled Nifti: rest_bold_nii = bold_dir / f'{sub}_resting.nii.gz'

    bold_dir = self.rest_dir / julia_sub / f'{julia_sub}_bold'

    bids_func_rest_filename = \
        f'sub-{sub}_ses-{self.session}_task-{self.task_name}_bold'

    dcm2bids(bold_dir, bids_func, bids_func_rest_filename)

  def copy_rest_t1w(self, sub, julia_sub):
    """Copies T1w nifti into BIDS anat/.

    Note: There is no DCM to be converted.

    """

    t1w = \
        self.rest_dir / julia_sub / 'T1' / \
        f'{julia_sub}_brain_Ret.nii.gz'

    bids = \
        self.out_dir / f'sub-{sub}' / f'ses-{self.session}' / 'anat' / \
        f'sub-{sub}_ses-{self.session}_T1w.nii.gz'

    shutil.copyfile(t1w, bids)

  def copy_rest_fmap(self, sub, julia_sub):
    """copies fieldmap files into fmap/ and create a sidecar."""

    fmap_dir = self.out_dir / f'sub-{sub}' / f'ses-{self.session}' / 'fmap'

    phase = \
        self.rest_dir / julia_sub / 'Phase_Image' / \
        f'{julia_sub}_Phase_rad_s.nii.gz'
    mag = \
        self.rest_dir / julia_sub / 'Magnitude_Image' / \
        f'{julia_sub}_Mag_fieldmap_brain.nii.gz'

    phase_bids = fmap_dir / f'sub-{sub}_ses-{self.session}_phasediff.nii.gz'
    mag_bids = fmap_dir / f'sub-{sub}_ses-{self.session}_magnitude1.nii.gz'
    sidecar_bids = fmap_dir / f'sub-{sub}_ses-{self.session}_phasediff.json'

    phase_sidecar = {
        'EchoTime1': None,
        'IntendedFor':
            (f'func/sub-{sub}_ses-'
             f'{self.session}_task-'
             f'{self.task_name}_bold.nii.gz')}

    with open(sidecar_bids, 'w') as f:
      json.dump(phase_sidecar, f, indent=2)

    shutil.copyfile(phase, phase_bids)
    shutil.copyfile(mag, mag_bids)

  def create_dataset_description(self):

    out_file = self.out_dir / f'dataset_description.json'

    dataset_description = {
        'Name': 'Julia 2018 Dataset',
        'BIDSVersion': '1.4.0',
        'DatasetType': 'raw',
        'Authors': ['Julia', 'Daphne']
    }

    with open(out_file, 'w') as f:
      json.dump(dataset_description, f, indent=2)

  def create_task_sidecar(self):

    task_sidecar = {
        'TaskName': self.task_name
    }

    out_file = self.out_dir / f'task-{self.task_name}_bold.json'

    with open(out_file, 'w') as f:
      json.dump(task_sidecar, f, indent=2)

  def is_valid(self):
    layout = BIDSLayout(self.out_dir, validate=True)
    layout.get
    validator = BIDSValidator()
    conditions = []
    for f in layout.files.keys():
      # bids-validator requires relative path, so fisrt converting abs to rel.
      path = f.replace(str(self.out_dir.absolute()), '')
      conditions.append(validator.is_bids(path))
    return all(conditions)
