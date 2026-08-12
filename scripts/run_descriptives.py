"""Descriptives the manuscript needs and can have today.

Three analyses, none of which is confirmatory and none of which should grow a
p-value later:

1. Demographics (manuscript §6.1, Appendix E item 9): sex and age by group,
   straight from `participants.tsv`. Item 9 calls the missing sex split "the
   most likely reviewer objection the current text cannot answer"; it is a
   one-table answer.

2. Group effect sizes for the three named couplings (§3.1): CON-SMN, FPN-SMN
   and CON-FPN partial correlation at network scale, as Cohen's d with a
   bootstrap interval. Descriptive by design — these features were *selected*
   because classifiers fitted to these same 32 participants weighted them, so
   testing them here would be circular (the manuscript says so explicitly).

3. The carryover check (§4.5): rest followed an attention task, so a group
   difference could be differential task carryover rather than trait. Carryover
   decays; a trait difference should not. Compare the effect in the early and
   late halves of the run: strong attenuation from early to late is evidence
   for carryover, stability is evidence against.

Usage:
    uv run python scripts/run_descriptives.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acnets.connectome import ExtraConnectivityMeasure
from acnets.pipeline import TimeseriesAggregator

DATA = Path('data/Julia2018/timeseries_dosenbach2010.nc5')
PARTICIPANTS = Path('data/julia2018/participants.tsv')
OUT = Path('outputs')

COUPLINGS = [('CON', 'SMN'), ('FPN', 'SMN'), ('CON', 'FPN')]


# ---------------------------------------------------------------- demographics

def demographics() -> dict:
    table = pd.read_csv(PARTICIPANTS, sep='\t')
    table['group'] = table['participant_id'].str[4:8]

    out: dict = {'n_total': len(table), 'groups': {}}
    for name, part in table.groupby('group'):
        out['groups'][name] = {
            'n': len(part),
            'sex': part['sex'].value_counts().to_dict(),
            'age_mean': float(part['age'].mean()),
            'age_sd': float(part['age'].std(ddof=1)),
            'age_range': [int(part['age'].min()), int(part['age'].max())],
        }
    return out


# ---------------------------------------------------------------- connectivity

def load_network_pcorr(dataset: xr.Dataset,
                       timepoints: slice | None = None) -> tuple[np.ndarray, list[str]]:
    """Network-scale partial correlation for all subjects.

    Fitted on everyone at once: this feeds descriptive effect sizes, not a
    classifier, so there is no fold to leak across.
    """
    working = dataset
    if timepoints is not None:
        working = dataset.isel(timepoint=timepoints)
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(working)
    values = aggregated['timeseries'].transpose('subject', 'timepoint', 'network').values
    networks = [str(n) for n in aggregated['network'].values]
    measure = ExtraConnectivityMeasure(kind='partial correlation', vectorize=False)
    return measure.fit(values).transform(values), networks


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with the pooled standard deviation."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float('nan')


def d_with_bootstrap(a: np.ndarray, b: np.ndarray,
                     n_boot: int = 5000,
                     random_state: int = 0) -> dict:
    """d(AVGP - NVGP) with a percentile bootstrap interval, resampled per group."""
    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = cohens_d(a[rng.integers(0, len(a), len(a))],
                            b[rng.integers(0, len(b), len(b))])
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return {'d': cohens_d(a, b), 'ci_low': float(low), 'ci_high': float(high),
            'excludes_zero': bool(low > 0 or high < 0)}


def coupling_effects(matrices: np.ndarray,
                     networks: list[str],
                     groups: np.ndarray) -> dict:
    index = {name: i for i, name in enumerate(networks)}
    avgp, nvgp = groups == 'AVGP', groups == 'NVGP'

    out = {}
    for x, y in COUPLINGS:
        values = matrices[:, index[x], index[y]]
        stats = d_with_bootstrap(values[avgp], values[nvgp])
        stats['mean_AVGP'] = float(values[avgp].mean())
        stats['mean_NVGP'] = float(values[nvgp].mean())
        stats['direction'] = ('AVGP > NVGP' if stats['mean_AVGP'] > stats['mean_NVGP']
                              else 'NVGP > AVGP')
        out[f'{x}-{y}'] = stats
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # ---- A1 demographics ---------------------------------------------------
    demo = demographics()
    print('demographics (manuscript 6.1, item 9):')
    for name, g in demo['groups'].items():
        sexes = ', '.join(f'{v} {k}' for k, v in sorted(g['sex'].items()))
        print(f'  {name}: n = {g["n"]}, {sexes}; '
              f'age {g["age_mean"]:.1f} +/- {g["age_sd"]:.1f} '
              f'({g["age_range"][0]}-{g["age_range"][1]})')

    dataset = xr.open_dataset(DATA, engine='h5netcdf').load()
    groups = np.array([str(s)[:4] for s in dataset['subject'].values])

    # ---- A4 effect sizes ---------------------------------------------------
    matrices, networks = load_network_pcorr(dataset)
    effects = coupling_effects(matrices, networks, groups)
    print('\ncoupling effect sizes, partial correlation at network scale '
          '(descriptive; 3.1):')
    for name, e in effects.items():
        print(f'  {name}: d = {e["d"]:+.2f} [{e["ci_low"]:+.2f}, {e["ci_high"]:+.2f}]  '
              f'{e["direction"]}  (AVGP {e["mean_AVGP"]:+.3f}, NVGP {e["mean_NVGP"]:+.3f})')

    # ---- A5 carryover ------------------------------------------------------
    n_timepoints = dataset.sizes['timepoint']
    half = n_timepoints // 2
    halves = {}
    for label, window in [('early', slice(0, half)), ('late', slice(half, None))]:
        m, nets = load_network_pcorr(dataset, timepoints=window)
        halves[label] = coupling_effects(m, nets, groups)

    print(f'\ncarryover check, early vs late half ({half} timepoints each; 4.5):')
    carryover = {}
    for name in ('CON-SMN', 'FPN-SMN'):
        early, late = halves['early'][name]['d'], halves['late'][name]['d']
        attenuation = (early - late) / abs(early) if early else float('nan')
        carryover[name] = {'early_d': early, 'late_d': late,
                           'attenuation': float(attenuation)}
        print(f'  {name}: early d = {early:+.2f}, late d = {late:+.2f}  '
              f'(attenuation {attenuation:+.0%})')
    print('  reading: carryover decays, a trait difference should not; strong '
          'early-to-late attenuation favours carryover.')

    payload = {'demographics': demo, 'coupling_effects': effects,
               'carryover': carryover,
               'note': ('descriptive only: the couplings were selected by '
                        'classifiers fitted to these same participants, so '
                        'inferential tests here would be circular')}
    (OUT / 'descriptives.json').write_text(json.dumps(payload, indent=2))
    print(f'\nwritten to {OUT / "descriptives.json"}')


if __name__ == '__main__':
    main()
