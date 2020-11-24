# %%
import logging
import pandas as pd

from pathlib import Path

bids_dir = Path('data') / 'julia2018' / 'bids'


def create_tfmri_events(sub, ses, bids_dir=bids_dir):

    logging.info(f'processing events for sub-{sub}_ses-{ses}...')

    beh_file = list((bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'beh').glob('*_trials.tsv'))[0]

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
        logging.info(f'saving run{r} events...')
        run_events = events.query('run == @r').drop(columns='run')
        run_events.to_csv(
            out_dir / f'sub-{sub}_ses-{ses}_task-attention_run-{r:02d}_events.tsv',
            index=False, float_format='%.3f')

    # cols = onset, duration, trial_type, stim_file, trial, block
    logging.info('done!')


create_tfmri_events(sub='NVGP01',
                    ses='2')

print('done!')
