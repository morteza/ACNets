# %% [markdown]
# converts julia2018 raw behavioral events into tidy trial data

from IPython.display import display
import pandas as pd
import numpy as np

in_file = 'data/julia2018_beh/NVGP14_A1_events.csv'

EVENTS = pd.read_csv(in_file)

EVENTS.query("type == 'end block 1'")


def get_block_index(event_type):
  if ' block ' in event_type:
    return event_type[-1]
  return None


def get_trial(events):
  """TODO: Given some events reconstruct trial as a single row."""
  pass


EVENTS['block_index'] = EVENTS.type.apply(get_block_index).ffill()
EVENTS['trial_index'] = (EVENTS.type.str.contains('cue: ')
                                    .replace(False, np.nan)
                                    .cumsum()
                                    .ffill())

# not clear which trial those 'missing arm' events belong to.
# TODO group and pivot EVENTS using get_trial
