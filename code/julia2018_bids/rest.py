import sys
import logging

from os import PathLike
from pathlib import Path
import shutil

import json

from dataclasses import dataclass

from utils import dcm2bids


@dataclass
class Julia2018RestBIDSifier():
  """Prepares BIDS-compatible resting state data."""

  in_dir: PathLike
  out_dir: PathLike
  session: str = 'rest'
  task_name: str = 'rest'

  def __post_init__(self) -> None:

    if isinstance(self.in_dir, str):
      self.in_dir = Path(self.in_dir)
    if isinstance(self.out_dir, str):
      self.out_dir = Path(self.out_dir)

    self.rest_dir = self.in_dir / 'Resting_State' / 'Resting_State'

    # TODO filter subjects with NEW suffix and append `acq` instead.
    self.subjects = [
        x.stem for x in self.rest_dir.iterdir() if x.is_dir()
    ]

    # TODO rename VGP/NVGP => AVGP/NVGP
    # self.subjects = [
    #     s.replace('VGP', 'AVG') if s.startswith('VGP') else s
    #     for s in self.subjects
    # ]

  def run(self):
    """Runs bidifier for all subjects."""

    for sub in self.subjects:
      self.create_folder_structure(sub)
      self.copy_rest_t1w(sub)
      self.prep_rest_bold(sub)
      self.copy_rest_fmap(sub)

    logging.info(
        'BIDS-ified Julia2018 Resting State (bold, T1w, fmap)'
    )

  def create_folder_structure(self, sub):
    """Creates initial folder structure for a given subject."""

    ses_dir = self.out_dir / f'sub-{sub}' / f'ses-{self.session}'
    (ses_dir / 'anat').mkdir(parents=True, exist_ok=True)
    (ses_dir / 'func').mkdir(parents=True, exist_ok=True)
    (ses_dir / 'fmap').mkdir(parents=True, exist_ok=True)

  def prep_rest_bold(self, sub):
    """convert BOLD DICOMs into nifti/sidecar, and move them into func/."""

    func_dir = self.out_dir / f'sub-{sub}' / f'ses-{self.session}' / 'func'

    # TODO: use heudiconv to convery DCM files

    bold_dir = self.rest_dir / sub / f'{sub}_bold'
    # rest_bold_nii = bold_dir / f'{sub}_resting.nii.gz'
    bids_rest_bold_filename = \
        f'sub-{sub}_ses-{self.session}_task-{self.task_name}_bold'

    dcm2bids(bold_dir, func_dir, bids_rest_bold_filename)

    pass

  def copy_rest_t1w(self, sub):
    """Copies T1w nifti into BIDS anat/.

    Note: There is no DCM to be converted.

    """

    anat_dir = self.out_dir / f'sub-{sub}' / f'ses-{self.session}' / 'anat'

    t1w = self.rest_dir / sub / 'T1' / f'{sub}_brain_Ret.nii.gz'
    bids = anat_dir / f'sub-{sub}_ses-{self.session}_T1w.nii.gz'

    shutil.copyfile(t1w, bids)

  def copy_rest_fmap(self, sub):
    """copies fieldmap files into fmap/ and create a sidecar."""

    fmap_dir = self.out_dir / f'sub-{sub}' / f'ses-{self.session}' / 'fmap'

    phase_filename = f'{sub}_Phase_rad_s.nii.gz'
    mag_filename = f'{sub}_Mag_fieldmap_brain.nii.gz'

    phase = self.rest_dir / sub / 'Phase_Image' / phase_filename
    mag = self.rest_dir / sub / 'Magnitude_Image' / mag_filename

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


# to test the Julia2018RestBIDSifier
if __name__ == "__main__":

  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

  in_dir = Path('data/julia2018_raw')
  out_dir = Path('data/julia2018_bids2')

  Julia2018RestBIDSifier(in_dir, out_dir).run()
