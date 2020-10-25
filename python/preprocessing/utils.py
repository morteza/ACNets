import os
from pathlib import Path
import json

try:
  from nipype.interfaces.dcm2nii import Dcm2niix
except ImportError:
  raise "dcm2bids() requires nipype and dcm2niix."


def dcm2bids(dcm_dir: Path,
             out_dir: Path,
             out_filename: str,
             overwrite: bool = True):
  """Converts raw DCM files to BIDS-compatible Nifti and sidecar.

  """

  out_path = out_dir / f'{out_filename}.nii.gz'

  if out_path.exists():
    if not overwrite:
      # warn if out_file exists; dcm2niix adds suffixes if
      # `out_file` exists, makes it incompatible to BIDS spec.
      raise Exception(
          f'[dcm2bids] Nifti already exists: "{out_filename}.nii.gz". '
          f'Delete it or enable overwrite flag in dcm2bids(...) function.'
      )

    # delete existing file if overwritting is allowed
    os.remove(out_path)

  converter = Dcm2niix()

  converter.inputs.source_dir = dcm_dir
  converter.inputs.compress = 'i'
  converter.inputs.output_dir = out_dir
  converter.inputs.out_filename = out_filename

  # DEBUG Path(out_dir).mkdir(parents=True, exist_ok=True)
  # DEBUG converter.inputs.compression = 5
  # DEBUG converter.inputs.has_private = False
  # DEBUG converter.cmdline

  return converter.run()


# TODO function to init BIDS folder structure and add readme description, etc

def init_bids(bids_dir: Path, dataset_name='Julia 2018 Dataset'):
  """Initializes folder structure and modality agnostic files."""

  bids_dir.mkdir(parents=True, exist_ok=True)

  # README
  with open(bids_dir / f'README', 'w') as f:
    f.write(f'# {dataset_name}')
    f.write('\n')

  # bidsignore
  with open(bids_dir / f'.bidsignore', 'w') as f:
    f.writelines([
        'sub-*/ses-A*/beh/*_trials.tsv',
        'sub-*/ses-Block*/beh/*_trials.tsv'
    ])

  # dataset_description
  dataset_description = {
      'Name': dataset_name,
      'BIDSVersion': '1.4.0',
      'DatasetType': 'raw',
      'Authors': ['Julia', 'Daphne']
  }

  with open(bids_dir / f'dataset_description.json', 'w') as f:
    json.dump(dataset_description, f, indent=2)


def init_ses(out_dir: Path, sub, ses):
  """Creates initial folder structure for a given subject and session."""

  ses_dir = out_dir / f'sub-{sub}' / f'ses-{ses}'
  (ses_dir / 'anat').mkdir(parents=True, exist_ok=True)
  (ses_dir / 'func').mkdir(parents=True, exist_ok=True)
  (ses_dir / 'fmap').mkdir(parents=True, exist_ok=True)
