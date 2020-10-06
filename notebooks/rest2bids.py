# %%
import logging
import os
from pathlib import Path
import shutil
import json

# Parameters. TODO: use a better approach to parametrize this script
rest_dir = Path('data/julia2018_raw/Resting_State')
bids_dir = Path('data/julia2018_BIDS/')

dataset_description = {
    'Name': 'Julia 2018 Dataset',
    'BIDSVersion': '1.4.0',
    'DatasetType': 'raw',
    'Authors': ['Julia', 'Daphne']
}
task_rest_sidecar = {
    'TaskName': 'rest'
}


def dcm2bids(dcm_dir: Path,
             out_dir: Path,
             out_filename: str,
             overwrite: bool = True):
  """Converts raw DCM files to BIDS-compatible Nifti and sidecar.

  """
  try:
    from nipype.interfaces.dcm2nii import Dcm2niix
  except ImportError:
    raise "dcm2bids() requires nipype and dcm2niix."

  out_filepath = out_dir / f'{out_filename}.nii.gz'

  if out_filepath.exists():
    if not overwrite:
      # warn if out_file exists; dcm2niix adds suffixes if
      # `out_file` exists, makes it incompatible to BIDS spec.
      raise Exception(
          f'[dcm2bids] Nifti already exists: "{out_filename}.nii.gz". '
          f'Delete it or enable overwrite flag in dcm2bids(...) function.'
      )

    # delete existing file if overwritting is allowed
    os.remove(out_filepath)

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


# 1. create BIDS project folder if not exist
bids_dir.mkdir(exist_ok=True)

# 2. create 'project_description.json'
with open(bids_dir / 'dataset_description.json', 'w') as f:
  json.dump(dataset_description, f, indent=2)
  f.write('\n')

# 3. create resting BOLD sidecar.
with open(bids_dir / 'task-rest_bold.json', 'w') as f:
  json.dump(task_rest_sidecar, f, indent=2)
  f.write('\n')

# 4. extract list of all participants
subjects = [x.stem for x in rest_dir.iterdir() if x.is_dir()]

# 5. loop over subjects and prepare their session data
for subj in subjects:

  logging.info(f'> Converting {subj} data into BIDS...')

  # 5.1. Initialize anat and func folder structures
  for mod in ['anat', 'func']:
    p = bids_dir / f'sub-{subj}' / 'ses-rest' / mod
    p.mkdir(parents=True, exist_ok=True)

  # 5.2. anat/T1w
  t1w = rest_dir / subj / 'T1' / f'{subj}_brain_Ret.nii.gz'
  bids_t1w = bids_dir / f'sub-{subj}' / 'ses-rest' / 'anat' / \
      f'sub-{subj}_ses-rest_T1w.nii.gz'
  shutil.copyfile(t1w, bids_t1w)

  # 5.3. func/rest_bold + sidecars
  rest_bold_dir = rest_dir / subj / f'{subj}_bold'
  rest_bold_nii = rest_bold_dir / f'{subj}_resting.nii.gz'
  bids_func_dir = bids_dir / f'sub-{subj}' / 'ses-rest' / 'func'
  bids_rest_bold_filename = f'sub-{subj}_ses-rest_task-rest_bold'

  # 5.4. Convert resting DICOMs to Nifti+sidecar and add them
  dcm2bids(rest_bold_dir, bids_func_dir, bids_rest_bold_filename)

  # DEBUG import the nifti that is already provided without DICOM conversion
  # DEBUG shutil.copyfile(rest_bold_nii,
  # DEBUG                 bids_func_dir / bids_rest_bold_filename)


# 5.5. Done!
logging.info(f'Done! BIDS-ready folder: {str(bids_dir.absolute())}')
