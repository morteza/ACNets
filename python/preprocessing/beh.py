from os import PathLike
from pathlib import Path

from dataclasses import dataclass
import re

import pandas as pd
import numpy as np

from bids import BIDSLayout, BIDSValidator


@dataclass
class Julia2018BehavioralPreprocessor():
  """Prepares BIDS-compatible behavioral data."""

  in_dir: PathLike
  bids_dir: PathLike
  overwrite: bool = False

  def __post_init__(self) -> None:

    self.in_dir = Path(self.in_dir)
    self.bids_dir = Path(self.bids_dir)

  @staticmethod
  def find_block_index(event_type):
    """Finds block index from event type."""
    if ' block ' in event_type:
      return event_type[-1]  # last char stores block index
    return None

  @staticmethod
  def events_to_trial(events):
    """Given some events, reconstructs trial as a single row."""
    cue_events = events.query('type.str.contains("cue: ")')
    stimulus_events = events.query('type.str.contains("target: ")')
    response_events = events.query('type.str.contains("response: ")')

    trial_index = int(events.trial_index.iloc[0])

    # Quality checks and warnings
    if len(cue_events) == 0:
      print('trial', trial_index, ': no cue')
    elif len(cue_events) > 1:
      print('trial', trial_index, ': multiple cues')

    # if len(response_events) == 0:
    #   print('trial', trial_index, ': no response')
    if len(response_events) > 1:
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
    cue_duration = None
    if len(stimulus_events) > 0:
      cue_duration = \
          stimulus_events.realTime.iloc[0] - cue_events.realTime.iloc[0]

    stimulus = None
    stimulus_ts = None
    stimulus_duration = None  # TODO calc stimulus_duration
    if len(stimulus_events) > 0:
      stimulus = \
          stimulus_events.type.apply(lambda s: s.split(' ')[1]).values[0]
      stimulus_ts = stimulus_events.realTime.iloc[0]

    response = None
    response_ts = None
    if len(response_events) > 0:
      response = \
          response_events.type.apply(lambda s: s.split(' ')[1]).values[0]
      response_ts = response_events.realTime.iloc[0]

    return pd.Series({
        'block_index': events.block_index.iloc[0],
        'trial_index': trial_index,
        'onset': None,     # BIDS
        'duration': None,  # BIDS
        'cue': cue,
        'stimulus': stimulus,
        'stimulus_timestamp': stimulus_ts,
        'stimulus_duration': stimulus_duration,
        'response': response,
        'cue_timestamp': cue_ts,
        'cue_duration': cue_duration,
        'response_timestamp': response_ts
    })

  def run(self):
    """Loops over all participants and prepare beh data."""

    # main loop over csv files
    for csv_file in self.in_dir.glob('**/*_events.csv'):

      sub, ses = re.search('([^_]+)_(.+)_events', csv_file.stem).groups()

      # fix BIDS error code 58 (TASK_NAME_CONTAIN_ILLEGAL_CHARACTER)
      ses = ses.replace('_', '').replace('-', '')

      group = re.search('([A-Z]+).*', sub).group(1)

      print(f'>>> parsing {sub} (Session {ses})...')

      TRIAL_PARAMS = pd.read_csv(str(csv_file).replace('_events', '_trials'))
      TRIAL_PARAMS['subject_id'] = sub
      TRIAL_PARAMS['group'] = group
      TRIAL_PARAMS['session'] = ses
      TRIAL_PARAMS['trial_index'] = TRIAL_PARAMS.index + 1
      TRIAL_PARAMS['has_missing_arms'] = TRIAL_PARAMS.missingArms > 0
      TRIAL_PARAMS.replace({
          'type': {
              'standardvalid': 'standard_valid',
              'standardinvalid': 'standard_invalid',
              'deviantvalid': 'distractor_valid',
              'deviantinvalid': 'distractor_invalid',
              'cue': 'catch'
          },
          'group': {
              'VGP': 'AVGP',
              'AVG': 'AVGP',
              'NAVGP': 'NVGP'
          }}, inplace=True)

      TRIAL_PARAMS.rename({
          'type': 'trial_type',
          'trialTime': 'trial_duration',
          'ITI': 'ITI',
          'SOA': 'SOA',
          'deviant': 'distractor',
          'missingArms': 'missing_arms_n'
      }, axis=1, inplace=True)

      # TODO extract stimulus_contrast
      TRIAL_PARAMS['stimulus_contrast'] = None
      TRIAL_PARAMS.drop(columns=['cue', 'gabor'], inplace=True)

      # now read and parse events; then extract trial data
      EVENTS = pd.read_csv(str(csv_file))

      EVENTS.sort_values(by='realTime', inplace=True)

      EVENTS['block_index'] = EVENTS.type.apply(self.find_block_index).ffill()

      EVENTS = EVENTS[~EVENTS.type.str.contains('trigger|block', regex=True)]

      # not clear which trial those 'missing arm' events must belong to.
      EVENTS['trial_index'] = (EVENTS.type.str.contains('cue: ')
                                          .replace(False, np.nan)
                                          .cumsum()
                                          .ffill())

      TRIALS = \
          EVENTS.groupby(['trial_index'], as_index=False). \
          apply(self.events_to_trial)

      TRIALS['rt'] = TRIALS.response_timestamp - TRIALS.stimulus_timestamp
      TRIALS['correct'] = \
          (TRIALS.stimulus == TRIALS.response) & (TRIALS.rt > 0)
      TRIALS['preparation_duration'] = \
          TRIALS.stimulus_timestamp - TRIALS.cue_timestamp

      TRIALS = TRIALS.merge(TRIAL_PARAMS, on='trial_index')

      # re-oreder columns to match UML diagram in the analysis plan
      TRIALS = TRIALS[[
          'onset',
          'duration',
          'subject_id',
          'group',
          'session',
          'block_index',
          'trial_index',
          'trial_type',
          'cue',
          'stimulus',
          'stimulus_contrast',
          'missing_arms_n',
          'response',
          'rt',
          'correct',
          'SOA',
          'ITI',
          'cue_duration',
          'preparation_duration',
          'stimulus_duration',
          'cue_timestamp',
          'stimulus_timestamp',
          'response_timestamp'
      ]]

      # VGP -> AVGP
      group = TRIALS.group.unique()[0]
      sub = group + sub[(-5 if sub.endswith('NEW') else -2):]

      beh_dir = self.bids_dir / f'sub-{sub}' / f'ses-{ses}' / 'beh'
      out_file = beh_dir / f'sub-{sub}_ses-{ses}_task-{ses}_trials.tsv'

      beh_dir.mkdir(parents=True, exist_ok=True)

      TRIALS.to_csv(out_file, sep='\t')  # .tsv

  def is_valid(self):
    layout = BIDSLayout(self.bids_dir, validate=True)
    layout.get
    validator = BIDSValidator()
    conditions = []
    for f in layout.files.keys():
      # bids-validator requires relative path, so fisrt converting abs to rel.
      path = f.replace(str(self.bids_dir.absolute()), '')
      conditions.append(validator.is_bids(path))
    return all(conditions)
