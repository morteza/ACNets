import logging
from dataclasses import dataclass, field
from typing import List
from os import PathLike
import shutil
from pathlib import Path
import json

from . import utils


@dataclass
class Julia2018TaskPreprocessor():
  """Prepares BIDS-compatible task-fMRI data."""

  in_dir: PathLike
  out_dir: PathLike
  subjects: List = field(default_factory=list)
  sessions: List = field(default_factory=lambda: ['A1', 'A2'])
  overwrite: bool = False

  task_dir = None  # to be initialized later in __post_init__

  def __post_init__(self) -> None:

    self.in_dir = Path(self.in_dir)
    self.out_dir = Path(self.out_dir)

    if (not self.overwrite) and self.out_dir.exists():
      raise Exception('Output directory exists. Either enable the `overwrite` flag '
                      'or use a different output dirctory.')

    self.task_dir = self.in_dir / 'RawDicom_A1_A2' / 'Attention'

    self.init_bids_subjects(self.subjects)

  def init_bids_subjects(self, julia2018_subjects):
    """init bids_subjects dict which maps bids subject id to julia2018 subject id."""

    if julia2018_subjects is None or len(julia2018_subjects) == 0:
      julia2018_subjects = [s.stem for s in self.task_dir.iterdir() if s.is_dir()]

    self.bids_subjects = {
        # rename VGP/NVGP => AVGP/NVGP
        s.replace('VGP', 'AVGP') if s.startswith('VGP') else s: s
        for s in julia2018_subjects
    }

  def run(self):
    utils.init_bids(self.out_dir)

    # Julia2018 uses the same names for tasks and sessions (A1 & A2).
    for s in self.sessions:
      self.create_task_sidecar(s)

    for sub, julia_sub in self.subjects.items():
      utils.init_ses(self.out_dir, sub, self.session)
      self.process_anat(sub, julia_sub)
      self.process_bold(sub, julia_sub)
      self.process_fmap(sub, julia_sub)

    logging.info('BIDS-ified Julia2018 task-fMRI (bold, T1w, fmap).')

  def process_bold(self, julia_subject, subject):
    """Copies BOLD Niftis into the BIDS structure."""
    # sub/NVGP01_brain_Att1.nii.gz
    # sub/NVGP01_brain_Att2.nii.gz

    for ses in self.sessions:
      bold = self.task_dir / julia_subject / f'{julia_subject}_brain_Att{ses}.nii.gz'

      bids = self.out_dir / f'sub-{subject}' / f'ses-{ses}' / 'func' / \
          f'sub-{subject}_ses-{ses}_task-{ses}_bold.nii.gz'

      shutil.copyfile(bold, bids)

  def process_anat(self, session):
    """converts anatomical DICOMs into Nifti and copies them into the BIDS structure."""
    pass

  def process_fmap(self, session):
    pass

  def process_beh(self, session):
    pass

  def process_events(self, session):
    pass

  def create_task_sidecar(self, task_name):
    """init task sidecar."""

    sidecar = {
        'TaskName': task_name
    }

    out_path = self.out_dir / f'task-{task_name}_bold.json'

    with open(out_path, 'w') as f:
      json.dump(sidecar, f, indent=2)
