# %%
from pathlib import Path
import json
import os
import stat
import subprocess
import pandas as pd


print('working directory:', os.getcwd())


raw_dir = Path('data/julia2018_raw/RawDicom_A1_A2/Attention')
bids_dir = Path('data/julia2018/bids')
subs = [p.stem for p in raw_dir.glob('*VGP*')]
subs = ['NVGP01']
sess = ['1', '2']


def create_tfmri_events(sub, ses, bids_dir=bids_dir):

    print(f'processing events for sub-{sub}_ses-{ses}...')

    beh_file = list((bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'beh').glob('*_beh.tsv'))[0]

    beh_df = pd.read_csv(beh_file, sep='\t')

    # block_offsets = beh.groupby('block_index').cue_onset.min()

    cue_onsets = beh_df.cue_onset_in_block
    cue_duration = 0.5

    stimulus_onsets = beh_df.stimulus_onset_in_block
    stimulus_duration = 0.1

    events = pd.DataFrame({
        'run': beh_df.block_index,
        'cue': cue_onsets,
        'stimulus': stimulus_onsets,
    })

    events = events.melt(id_vars='run', var_name='event_type', value_name='onset').sort_values(['run', 'onset'])

    events['duration'] = events.event_type.apply(
        lambda t: cue_duration if t == 'cue' else stimulus_duration
    )

    out_dir = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in events['run'].unique():
        print(f'storing run{r} events...')
        run_events = events.query('run == @r').drop(columns='run')
        run_events = run_events[['onset', 'duration', 'event_type']]
        run_events.to_csv(
            out_dir / f'sub-{sub}_ses-{ses}_task-attention_run-{r:02d}_events.tsv',
            sep='\t', na_rep='n/a', index=False, float_format='%.3f')

    # TODO cols = onset, duration, trial_type, event_type, stim_file, trial, block


def fix_fmap_intended_for(sub, ses, bids_dir=bids_dir):
  fmap_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'fmap'
  func_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'func'

  # TODO there are two fieldmaps, each only intended for a subset of BOLD scans.
  # TODO the second fieldmap scan is intended for func/run-05,run-06,run-07,run-08.
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

    # res = subprocess.run(cmd.split(' '), stderr=subprocess.PIPE, text=True, cwd=os.getcwd())

    # print(res.stderr)
    create_tfmri_events(sub, ses)
    # fix_fmap_intended_for(sub, ses)

print('FINISHED!')
