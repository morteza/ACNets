# %%
import pandas as pd

from pathlib import Path
data_dir = Path('data') / 'julia2018'

# load trials data
sub = 'NVGP01'
ses = '1'
beh_file = list((data_dir / 'bids' / f'sub-{sub}' / f'ses-{ses}' / 'beh').glob('*_trials.tsv'))[0]

beh = pd.read_csv(beh_file, sep='\t')

block_offsets = beh.groupby('block_index').cue_onset.min()

cue_onsets = beh.cue_onset_in_block
cue_duration = 0.5

stimulus_onsets = beh.stimulus_onset_in_block
stimulus_duration = 0.1

events = pd.DataFrame({
    'run': beh.block_index,
    'cue': cue_onsets,
    'stimulus': stimulus_onsets,
})

events = events.melt(id_vars='run', var_name='event_type', value_name='onset').sort_values(['run', 'onset'])

events['duration'] = events.event_type.apply(lambda t: cue_duration if t == 'cue' else stimulus_duration)


data_dir / 'bids' / f'sub-{sub}' / f'ses-{ses}' / 'func' / f'sub-{sub}_ses-{ses}_task-A{ses}_events.tsv'
