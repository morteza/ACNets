# %%

""" Note: for convenience, resting-state is considered session 3, which is
          compatible with datatimes of the imaging files."""

from pathlib import Path
import shutil
import json

rest_dir = Path('data/julia2018_raw/Resting_State')
bids_dir = Path('data/Julia2018BIDS/')
dataset_description = {
    'Name': 'Julia2008 Dataset',
    'BIDSVersion': '1.4.0',
    'DatasetType': 'raw',
    'Authors': ['Julia', 'Daphne']
}


bids_dir.mkdir(exist_ok=True)


# (bids_dir / 'dataset_description.json').touch(exist_ok=True)
with open(bids_dir / 'dataset_description.json', 'w') as f:
  json.dump(dataset_description, f, indent=2)
  f.write('\n')


subjects = [x.stem for x in rest_dir.iterdir() if x.is_dir()]

for subj in subjects:

  print(f'> Converting {subj} data into BIDS...')

  # Initialize folder structure
  for mod in ['anat', 'func']:
    p = bids_dir / f'sub-{subj}' / 'ses-3' / mod
    p.mkdir(parents=True, exist_ok=True)

  # anat/T1w
  t1w = rest_dir / subj / 'T1' / f'{subj}_brain_Ret.nii.gz'
  bids_t1w = bids_dir / f'sub-{subj}' / 'ses-3' / 'anat' / \
      f'sub-{subj}_ses-3_T1w.nii.gz'
  shutil.copyfile(t1w, bids_t1w)

  # func/rest-bold
  rest_bold = rest_dir / subj / f'{subj}_bold' / f'{subj}_resting.nii.gz'
  bids_rest_bold = bids_dir / f'sub-{subj}' / 'ses-3' / 'func' / \
      f'sub-{subj}_ses-3_task-rest_bold.nii.gz'
  shutil.copyfile(rest_bold, bids_rest_bold)

  # func/rest-bold sidecar
  # TODO: use nibabeldicom2nifti instead to convert DICOMs into Nifti+Sidecars

  rest_bold_sidecar = {
      'TaskName': 'rest',
      'RepetitionTime': 3.0
  }

  rest_bold_sidecar_path = bids_dir / f'sub-{subj}' / 'ses-3' / 'func' / \
      f'sub-{subj}_ses-3_task-rest_bold.json'

  with open(rest_bold_sidecar_path, 'w') as f:
    json.dump(rest_bold_sidecar, f, indent=2)
    f.write('\n')


print('Done!')
print(f'BIDS-ready folder: {str(bids_dir.absolute())}')
