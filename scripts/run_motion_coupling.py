"""Does residual head motion explain the features the classifiers rely on?

`run_motion_control.py` establishes two uncomfortable facts: a classifier given
*only* motion summaries reaches 0.61, comparable to the best connectivity
specification, and AVGPs move more than NVGPs (framewise-displacement SD,
d = +0.89). Standard confound regression -- nilearn's 'simple' strategy, six
realignment parameters plus WM/CSF -- is known not to remove the subject-level
relationship between motion and connectivity (Power et al. 2012; Satterthwaite
et al. 2012), so that relationship has to be tested directly rather than
assumed away.

The specific worry is not generic. Head motion inflates apparent coupling most
in sensorimotor cortex, and the feature the multiverse agrees on is CON-SMN.
A motion account therefore predicts exactly the observed attribution, which
makes this the sharpest alternative explanation the paper faces.

Three questions, per connectivity statistic, on the network-scale features:

1. Does subject-level motion correlate with the CON-SMN and FPN-SMN edges?
2. Does the group difference in those edges survive partialling motion out?
3. Does the group difference in *any* edge survive it?

Nothing here is cross-validated: these are descriptive subject-level
associations over all 32 participants, which is the right level for a confound
check and the wrong level for a performance claim.

    uv run python scripts/run_motion_coupling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.analysis.multiscale import partial_correlation
from acnets.connectome import ExtraConnectivityMeasure
from acnets.multiverse.runner import edge_names, vectorize
from run_agreement import RUNNABLE_STATISTICS, aggregate, group_labels, load_atlas
from run_motion_control import load as load_motion

OUT = Path('outputs/controls')

# the two features the multiverse agrees on (section 3.1)
FOCUS = ('CON ↔ SMN', 'FPN ↔ SMN')

# one scalar summary of motion per participant: the channel with the largest
# group difference, chosen in run_motion_control and named here so the choice
# is visible rather than buried
MOTION_CHANNEL = 'framewise_displacement_sd'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = load_atlas('dosenbach2010')
    labels, _ = group_labels(dataset)
    all_timeseries, all_nodes = aggregate(dataset, 0)
    timeseries, nodes = all_timeseries['network'], all_nodes['network']
    motion_table, motion_y = load_motion()

    if not np.array_equal(labels, motion_y):
        raise SystemExit('participant order differs between the two loaders')

    motion = motion_table[MOTION_CHANNEL].to_numpy(dtype=float)
    # AVGP = 0, so a negative t means AVGPs are higher
    group = labels.astype(float)
    print(f'{len(labels)} participants, motion summary = {MOTION_CHANNEL}\n')

    payload: dict[str, dict] = {}
    for statistic in RUNNABLE_STATISTICS:
        # fit/transform rather than fit_transform: the latter reaches nilearn's
        # implementation, which does not know the two extra statistics
        measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
        measure.fit(timeseries)
        matrices = measure.transform(timeseries)
        asymmetric = statistic in ('chatterjee', 'transfer_entropy')
        x = vectorize(matrices, asymmetric)
        names = edge_names(nodes, asymmetric)

        rows = []
        for j, name in enumerate(names):
            feature = x[:, j]
            # a Chatterjee matrix has xi(a, a) = 1 on the diagonal, so those
            # columns are constant and every correlation with them undefined
            if np.ptp(feature) < 1e-12:
                continue
            r_motion, p_motion = stats.pearsonr(feature, motion)
            t_group, p_group = stats.ttest_ind(feature[labels == 0], feature[labels == 1],
                                               equal_var=False)
            r_partial, p_partial = partial_correlation(feature, group, motion)
            rows.append({'edge': name,
                         'r_with_motion': float(r_motion), 'p_motion': float(p_motion),
                         't_group': float(t_group), 'p_group': float(p_group),
                         'r_group_given_motion': r_partial,
                         'p_group_given_motion': p_partial})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / f'motion_coupling_{statistic.replace(" ", "_")}.csv',
                     index=False)

        focus = frame[frame['edge'].isin(FOCUS)]
        print(f'{statistic}:')
        for _, r in focus.iterrows():
            print(f'  {r["edge"]:12s} motion r = {r["r_with_motion"]:+.3f} '
                  f'(p = {r["p_motion"]:.3f})   group p = {r["p_group"]:.3f}   '
                  f'group|motion p = {r["p_group_given_motion"]:.3f}')
        n_group = int((frame['p_group'] < 0.05).sum())
        n_partial = int((frame['p_group_given_motion'] < 0.05).sum())
        n_motion = int((frame['p_motion'] < 0.05).sum())
        print(f'  edges nominally significant: {n_group} by group, '
              f'{n_partial} by group given motion, {n_motion} correlated with motion '
              f'(of {len(frame)})\n')

        payload[statistic] = {
            'n_edges': len(frame),
            'n_group_significant': n_group,
            'n_group_significant_given_motion': n_partial,
            'n_correlated_with_motion': n_motion,
            'focus': {r['edge']: {k: r[k] for k in r.index if k != 'edge'}
                      for _, r in focus.iterrows()},
        }

    (OUT / 'motion_coupling.json').write_text(
        json.dumps({'motion_channel': MOTION_CHANNEL,
                    'n_subjects': len(labels),
                    'statistics': payload}, indent=2, default=float))
    print(f'written to {OUT}/')


if __name__ == '__main__':
    main()
