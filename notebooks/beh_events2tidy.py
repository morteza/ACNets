# %%
# converts julia2018 raw behavioral events into tidy trial data
# NOTE: this is a work-in-progress.


import re
import pandas as pd
import numpy as np

from pathlib import Path


input_dir = 'data/julia2018_beh/'
output_dir = 'data/julia2018_beh_tidy/'

Path(output_dir).mkdir(exist_ok=True)


def get_block_index(event_type):
  if ' block ' in event_type:
    return event_type[-1]  # last char stores block index
  return None


def get_trial(events):
  """Given some events reconstruct trial as a single row."""
  cue_events = events.query('type.str.contains("cue: ")')
  stimulus_events = events.query('type.str.contains("target: ")')
  response_events = events.query('type.str.contains("response: ")')

  trial_index = int(events.trial_index.iloc[0])

  # Quality checks and warnings
  if len(cue_events) == 0:
    print('trial', trial_index, ': no cue')
  elif len(cue_events) > 1:
    print('trial', trial_index, ': multiple cues')

  if len(response_events) == 0:
    print('trial', trial_index, ': no response')
  elif len(response_events) > 1:
    response_events = response_events.iloc[[0]]
    print('trial', trial_index, ': multiple responses')

  if len(stimulus_events) == 0:
    print('trial', trial_index, ': no stimulus')
    # workaround for NVGP037_A2 data issues
    stimulus_events = cue_events.copy()
    stimulus_events[['realTime', 'type']] = (np.nan, ' ')
  elif len(stimulus_events) > 1:
    print('trial', trial_index, ': multiple stimuli')

  # TODO quality checks to verify number of missing arms

  cue = cue_events.type.apply(lambda s: s.split(' ')[1]).values[0]
  cue_ts = cue_events.realTime.iloc[0]

  stimulus = \
      stimulus_events.type.apply(lambda s: s.split(' ')[1]).values[0] \
      if len(stimulus_events) > 0 else None
  stimulus_ts = stimulus_events.realTime.iloc[0]

  response = None
  response_ts = None
  if len(response_events) > 0:
    response = response_events.type.apply(lambda s: s.split(' ')[1]).values[0]
    response_ts = response_events.realTime.iloc[0]

  return pd.Series({
      'block_index': events.block_index.iloc[0],
      'trial_index': trial_index,
      'cue': cue,
      'stimulus': stimulus,
      'response': response,
      'cue_timestamp': cue_ts,
      'stimulus_timestamp': stimulus_ts,
      'response_timestamp': response_ts
  })


for csv_file in Path(input_dir).glob('**/NVGP07_A1_events.csv'):

  participant_id, session_index = re.search('([^_]+)_(.+)_events',
                                            csv_file.stem,
                                            re.IGNORECASE).groups()

  print(f'parsing {participant_id} (Session {session_index})...')

  output_path = Path(output_dir).joinpath(csv_file.stem + '.csv')

  TRIAL_PARAMS = pd.read_csv(str(csv_file).replace('_events', '_trials'))
  TRIAL_PARAMS['trial_index'] = TRIAL_PARAMS.index + 1

  EVENTS = pd.read_csv(str(csv_file))
  EVENTS.sort_values(by='realTime', inplace=True)

  EVENTS['block_index'] = EVENTS.type.apply(get_block_index).ffill()

  EVENTS = EVENTS[~EVENTS.type.str.contains('trigger|block', regex=True)]

  # not clear which trial those 'missing arm' events must belong to.
  EVENTS['trial_index'] = (EVENTS.type.str.contains('cue: ')
                                      .replace(False, np.nan)
                                      .cumsum()
                                      .ffill())

  TRIALS = EVENTS.groupby(['trial_index'], as_index=False).apply(get_trial)
  TRIALS['rt'] = TRIALS.response_timestamp - TRIALS.stimulus_timestamp
  TRIALS['correct'] = (TRIALS.stimulus == TRIALS.response) & (TRIALS.rt > 0)

  TRIALS = TRIALS.merge(TRIAL_PARAMS, on='trial_index')
  TRIALS.to_csv(output_path)
