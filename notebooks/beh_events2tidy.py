# %%
# converts julia2018 raw behavioral events into tidy trial data
# NOTE: this is a work-in-progress.


import pandas as pd
import numpy as np

from pathlib import Path


input_dir = 'data/julia2018_beh/'
output_dir = 'data/julia2018_beh_tidy/'

Path(output_dir).mkdir(exist_ok=True)


def get_block_index(event_type):
  if ' block ' in event_type:
    return event_type[-1]
  return None


def get_trial(events):
  """Given some events reconstruct trial as a single row."""
  cue_event = events.query('type.str.contains("cue: ")')
  stimulus_event = events.query('type.str.contains("target: ")')
  response_event = events.query('type.str.contains("response: ")')
  # RT, correct, timestamp
  stimuli = stimulus_event.type.apply(lambda s: s.split(' ')[1]).values
  responses = response_event.type.apply(lambda s: s.split(' ')[1]).values
  return pd.Series({
      'cue': cue_event.type.apply(lambda s: s.split(' ')[1]).values[0],
      'target': None,
      'stimulus': stimuli[0] if len(stimuli) > 0 else None,
      'response': responses[0] if len(responses) > 0 else None,
      'cue_timestamp': cue_event.blockTime.values[0],  # block timestamp
      'stimulus_onset': (stimulus_event.
                         blockTime.values[0] - cue_event.blockTime.values[0]),
      'rt': (response_event.
             blockTime.
             values[0] - stimulus_event.
             blockTime.values[0] if len(responses) > 0 else None)
  })


for csv_file in Path(input_dir).glob('**/*_events.csv'):
  print(f'parsing {csv_file.stem}...')

  EVENTS = pd.read_csv(str(csv_file))

  EVENTS['block_index'] = EVENTS.type.apply(get_block_index).ffill()
  EVENTS['trial_index'] = (EVENTS.type.str.contains('cue: ')
                                      .replace(False, np.nan)
                                      .cumsum()
                                      .ffill())
  # not clear which trial those 'missing arm' events belong to.

  TRIALS = EVENTS.groupby(['block_index', 'trial_index']).apply(get_trial)
  TRIALS['correct'] = (TRIALS.stimulus == TRIALS.response) & ~TRIALS.rt.isna()

  output_path = Path(output_dir).joinpath(csv_file.stem + '_tidy.csv')
  TRIALS.to_csv(output_path)

# TODO: Fails for NVGP37_A2_Block1_events (due to missing stimuli)
