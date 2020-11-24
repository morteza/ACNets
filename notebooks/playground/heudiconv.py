# %%
from pathlib import Path
import json
import os
import stat
import subprocess


print('working directory:', os.getcwd())


raw_dir = Path('data/julia2018_raw/RawDicom_A1_A2/Attention')
bids_dir = Path('data/julia2018_tfmri_bids')
subs = [p.stem for p in raw_dir.glob('*VGP*')]
subs = ['NVGP01']
sess = ['1', '2']


def fix_fmap_intended_for(sub, ses, bids_dir=bids_dir):
  fmap_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'fmap'
  func_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'func'

  fmap_sidecars = fmap_path.glob('*.json')
  intended_for = [
      str(ni.relative_to(bids_dir / f'sub-{sub}'))
      for ni in func_path.glob('*.nii.gz')]

  for file in fmap_sidecars:
    print(f'updating IntendedFor in {file}...')
    os.chmod(file, stat.S_IRUSR | stat.S_IWUSR)
    with open(file) as f:
      data = json.load(f)
      data['IntendedFor'] = intended_for
    with open(file, 'w') as f:
      json.dump(data, f, indent=2, sort_keys=True)


for ses in sess:
  for sub in subs:
    cmd = ('heudiconv'
           ' --bids'
           ' -f python/preprocessing/heuristic.py'
           f' --files {str(raw_dir)}/{sub}/RawDICOM/Attention{ses}'
           f' -s {sub}'
           f' -ss {ses}'
           f' -o {str(bids_dir)}'
           ' -c dcm2niix')
    print(cmd)

    res = subprocess.run(cmd.split(' '), stderr=subprocess.PIPE, text=True, cwd=os.getcwd())

    print(res.stderr)
    fix_fmap_intended_for(sub, ses)

print('FINISHED!')
