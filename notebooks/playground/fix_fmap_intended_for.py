# %%
import logging
import json
from pathlib import Path
import os
import stat

bids_dir = Path('data') / 'julia2018_tfmri_bids'


subs = ['NVGP01']
sess = ['1']

for sub in subs:
  for ses in sess:
    fmap_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'fmap'
    func_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'func'

    fmap_sidecars = fmap_path.glob('*.json')
    intended_for = [
        str(ni.relative_to(bids_dir / f'sub-{sub}' / f'ses-{ses}'))
        for ni in func_path.glob('*.nii.gz')]

    for file in fmap_sidecars:
      print(f'updating IntendedFor in {file}...')
      os.chmod(file, stat.S_IRUSR | stat.S_IWUSR)
      with open(file) as f:
        data = json.load(f)
        data['IntendedFor'] = intended_for
      with open(file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
      print('done!')
