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

    BEH = pd.read_csv(beh_file, sep='\t')

    # block_offsets = beh.groupby('block_index').cue_onset.min()

    cue_duration = 0.5
    stimulus_duration = 0.1

    events = pd.DataFrame({
        'run': BEH.block_index,
        'cue': BEH.cue_onset_in_block,
        'stimulus': BEH.stimulus_onset_in_block,
        'trial_type': BEH.trial_type,
        'cue_file': BEH.cue,
        'stim_file': BEH.stimulus,
        'trial': BEH.trial_index
    })

    events = events.melt(id_vars=['run', 'trial', 'trial_type', 'cue_file', 'stim_file'],
                         var_name='event_type',
                         value_name='onset').sort_values(['run', 'onset'])

    events['duration'] = events.event_type.apply(
        lambda t: cue_duration if t == 'cue' else stimulus_duration
    )

    events.loc[events['event_type'] == 'cue', 'stim_file'] = events['cue_file']

    out_dir = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in events['run'].unique():
        print(f'storing run{r} events...')
        run_events = events.query('run == @r').drop(columns='run')
        run_events = run_events[['onset', 'duration', 'trial_type', 'event_type', 'trial', 'stim_file']]
        run_events.to_csv(
            out_dir / f'sub-{sub}_ses-{ses}_task-attention_run-{r:02d}_events.tsv',
            sep='\t', na_rep='n/a', index=False, float_format='%.3f')


def fix_fmap_intended_for(sub, ses, bids_dir=bids_dir):
  """Uses scans sidecar to detect proper intendedFor field for fmap sidecars."""

  ses_path = bids_dir / f'sub-{sub}' / f'ses-{ses}'
  fmap_path = bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'fmap'

  # load *_scan.tsv and processes fmaps
  SCANS = pd.read_csv(ses_path / f'sub-{sub}_ses-{ses}_scans.tsv', sep='\t')
  fmap_modalities = ['magnitude1', 'magnitude2', 'phasediff']

  for mod in fmap_modalities:
    SCANS.loc[SCANS['filename'].str.contains(mod), mod] = SCANS['filename']
    SCANS[mod].ffill(inplace=True)

  SCANS = SCANS[SCANS['filename'].str.startswith('func/')]
  SCANS = SCANS.melt(id_vars='filename', value_name='fmap', value_vars=fmap_modalities)

  # now put an IntendedFor field in the fmap sidecar jsons
  for file in fmap_path.glob('*.json'):
    print(f'updating IntendedFor in {file}...')
    intended_for = [f'ses-{ses}/{func}' for func in SCANS.query('fmap.str.contains(@file.stem)')['filename'].tolist()]

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
    try:
      create_tfmri_events(sub, ses)
    except Exception:
      print(f'cannot create events for sub-{sub}_ses-{ses}')
    fix_fmap_intended_for(sub, ses)

print('FINISHED!')
