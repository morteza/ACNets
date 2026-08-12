"""Brain-behaviour association: do the diagnostic couplings track performance?

Manuscript §3.3, Appendix E item 6b — load-bearing under the transfer framing.
The claim "reconfigured control-to-sensorimotor coupling supports broad
transfer" should bear some relation to measured attentional performance, and
this is where that relation is tested.

The design point that matters (and the reason `pooled_and_within` exists): a
pooled correlation across AVGPs and NVGPs can be produced entirely by the group
difference. Only a within-group association says the coupling tracks ability.
Both are always reported, with the verdict stated by the module.

Behaviour comes from the BIDS `beh/` files that `preprocessing/beh.py` produced
(restored from DVC; byte-identical to the originals). Performance measures per
subject, pooled over both sessions:

    accuracy  proportion correct on standard (non-catch) trials
    rt        median response time on correct standard trials, seconds
    ies       inverse efficiency score, rt / accuracy (lower is better)

Connectivity features are the two couplings §3.1 reports (CON-SMN, FPN-SMN),
partial correlation at network scale, computed on all 32 subjects at once —
descriptive, no folds, same as `run_descriptives.py`.

Usage:
    uv run python scripts/run_brain_behavior.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.analysis.brain_behavior import feature_performance_table
from run_descriptives import COUPLINGS, load_network_pcorr

BIDS = Path('data/julia2018')
DATA = Path('data/Julia2018/timeseries_dosenbach2010.nc5')
OUT = Path('outputs')

FEATURES = [f'{x}-{y}' for x, y in COUPLINGS if y == 'SMN']  # CON-SMN, FPN-SMN

# The seven replacement participants carry different identifiers in the two
# record systems: the imaging side suffixes the original number with 'NEW',
# the behavioural side renumbers it by +20. They are the same seven people,
# confirmed by the investigators.
#
# This mattered. An inner join on the raw identifiers silently dropped all
# seven, leaving 25 of 32 participants -- and the shortfall was then written up
# as those participants having no task records, which is not what happened.
# Five of the seven are AVGP, so the loss was also not balanced across groups
# (11 AVGP / 14 NVGP against the true 16 / 16).
#
# Written out rather than derived from the +20 pattern so that a wrong pairing
# is visible in review rather than hidden in arithmetic, and asserted below so
# a change to either record system fails loudly.
REPLACEMENT_IDS = {
    'AVGP32': 'AVGP12NEW',
    'AVGP33': 'AVGP13NEW',
    'AVGP34': 'AVGP14NEW',
    'AVGP36': 'AVGP16NEW',
    'AVGP37': 'AVGP17NEW',
    'NVGP37': 'NVGP17NEW',
    'NVGP39': 'NVGP19NEW',
}


def behavioural_performance() -> pd.DataFrame:
    """Per-subject performance, indexed by canonical BIDS subject id.

    The id comes from the file path, not from the `subject_id` column inside
    the file: the column drops the group prefix ('VGP01' for sub-AVGP01), and
    the path is what everything else in the project is keyed by.
    """
    rows = []
    for path in sorted(BIDS.glob('sub-*/ses-*/beh/*_task-attention_beh.tsv')):
        subject = path.parts[2].removeprefix('sub-')
        trials = pd.read_csv(path, sep='\t', na_values=['n/a'])
        trials['subject'] = subject
        rows.append(trials)
    if not rows:
        raise SystemExit(f'no behavioural files under {BIDS}')
    trials = pd.concat(rows, ignore_index=True)

    # catch trials expect no response; they do not enter accuracy or RT
    standard = trials[trials['trial_type'] != 'catch'].copy()
    standard['correct'] = standard['correct'].astype(str) == 'True'

    def summarise(part: pd.DataFrame) -> pd.Series:
        accuracy = float(part['correct'].mean())
        correct_rt = part.loc[part['correct'], 'response_time'].astype(float)
        rt = float(correct_rt.median()) if len(correct_rt) else float('nan')
        return pd.Series({
            'n_trials': len(part),
            'accuracy': accuracy,
            'rt': rt,
            'ies': rt / accuracy if accuracy > 0 else float('nan'),
        })

    performance = standard.groupby('subject').apply(summarise, include_groups=False)
    return to_imaging_ids(performance)


def to_imaging_ids(performance: pd.DataFrame) -> pd.DataFrame:
    """Re-key behavioural records onto the identifiers the imaging data uses.

    Every other table in this project is keyed by the imaging identifier, so
    canonicalising here fixes each consumer at once rather than requiring every
    join to know about the renaming.
    """
    renamed = performance.rename(index=REPLACEMENT_IDS)
    if len(set(renamed.index)) != len(renamed.index):
        raise SystemExit('replacement id mapping collided with an existing id')
    return renamed


def connectivity_features() -> pd.DataFrame:
    dataset = xr.open_dataset(DATA, engine='h5netcdf').load()
    matrices, networks = load_network_pcorr(dataset)
    index = {name: i for i, name in enumerate(networks)}
    subjects = [str(s) for s in dataset['subject'].values]

    frame = pd.DataFrame(index=pd.Index(subjects, name='subject'))
    for x, y in COUPLINGS:
        if y != 'SMN':
            continue
        frame[f'{x}-{y}'] = matrices[:, index[x], index[y]]
    return frame


def main() -> None:
    OUT.mkdir(exist_ok=True)

    performance = behavioural_performance()
    features = connectivity_features()

    joined = features.join(performance, how='inner')
    print(f'behaviour: {len(performance)} subjects; imaging: {len(features)}; '
          f'joined: {len(joined)}')
    missing = sorted(set(features.index) - set(performance.index))
    if missing:
        print(f'  imaging subjects without behaviour: {missing}')
    # The join used to drop the seven replacement participants through an
    # identifier mismatch (see REPLACEMENT_IDS). Failing loudly is better than
    # quietly analysing a subset again.
    if len(joined) < len(features):
        raise SystemExit(
            f'{len(features) - len(joined)} imaging subjects lost in the join: '
            f'{missing}. Check REPLACEMENT_IDS before proceeding.')
    joined['group'] = [s[:4] for s in joined.index]

    print('\nperformance by group:')
    for name, part in joined.groupby('group'):
        print(f'  {name}: accuracy {part["accuracy"].mean():.3f} '
              f'+/- {part["accuracy"].std(ddof=1):.3f}, '
              f'median RT {part["rt"].mean():.3f} s, '
              f'IES {part["ies"].mean():.3f}')

    table = feature_performance_table(
        {name: joined[name].to_numpy() for name in FEATURES},
        {name: joined[name].to_numpy() for name in ('accuracy', 'rt', 'ies')},
        joined['group'].to_numpy())

    print('\nassociations (r with 95% bootstrap CI; within-group is the '
          'informative row):')
    for (feature, measure), part in table.groupby(['feature', 'measure'],
                                                  sort=False):
        print(f'  {feature} x {measure}')
        for _, r in part.iterrows():
            marker = '  <-- excludes zero' if r['excludes_zero'] else ''
            print(f'    {r["label"]:13s} r = {r["r"]:+.3f} '
                  f'[{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]  n = {r["n"]:.0f}'
                  f'{marker}')

    # per-pair verdicts, from the module's own interpretation logic
    from acnets.analysis.brain_behavior import pooled_and_within
    verdicts = {}
    for feature in FEATURES:
        for measure in ('accuracy', 'rt', 'ies'):
            frame = pooled_and_within(joined[feature].to_numpy(),
                                      joined[measure].to_numpy(),
                                      joined['group'].to_numpy())
            verdicts[f'{feature} x {measure}'] = frame.attrs['interpretation']

    print('\nverdicts:')
    seen = set()
    for pair, verdict in verdicts.items():
        if verdict not in seen:
            print(f'  {verdict}')
            seen.add(verdict)
        print(f'    - {pair}')

    payload = {
        'n_joined': len(joined),
        'performance_by_group': {
            name: {'accuracy': float(part['accuracy'].mean()),
                   'rt': float(part['rt'].mean()),
                   'ies': float(part['ies'].mean())}
            for name, part in joined.groupby('group')},
        'associations': table.drop(columns=[]).to_dict(orient='records'),
        'verdicts': verdicts,
        'note': table.attrs['note'],
    }
    (OUT / 'brain_behavior.json').write_text(json.dumps(payload, indent=2))
    print(f'\nwritten to {OUT / "brain_behavior.json"}')


if __name__ == '__main__':
    main()
