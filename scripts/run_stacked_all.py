"""One model over every participant and every modality, with one corrected accuracy.

The obvious way to combine four modalities is to intersect them, and that is what
`run_full_multimodal.py` does: 833 cells on the 25 participants who have
everything. It answers a real question and it throws away 54 of 79 people,
including all 47 of gong2015, because modality availability in this project is
nested inside dataset:

    structural GMV   79/79   (47 gong2015 + 32 Julia2018)
    functional FC    32/79   (Julia2018 only)
    behaviour        25/79   (Julia2018 only; 7 replacements have no records)

So "all the data at once" and "all the participants at once" are different
analyses, and the intersection gives the second only by abandoning the first.

This script takes the other branch. Each modality gets a base learner trained on
whoever has it, out-of-fold base scores become the features of a meta-learner,
and the meta-learner is told which modalities each participant actually has. A
gong2015 participant contributes a structural score and two missing-indicators; a
Julia2018 participant with behavioural records contributes three scores. Every one
of the 79 is predicted and scored, so the headline accuracy is over the whole
pooled sample rather than over its richest corner.

**What this can and cannot buy.** It cannot manufacture functional data for
gong2015, so the functional path is still estimated from 32 people and its
contribution to the meta-learner is bounded by that. What it does buy is that the
structural path is estimated from 79 rather than 32, and that a single number with
a single family-wise null describes the whole design instead of one corner of it.

**Two things are load-bearing and are reported rather than assumed.**

The label mapping. gong2015's "non-experts" are amateur action gamers, not
non-players, and equating them with Julia2018's NVGPs assumes exposure is one
dimension along which amateur and non-player sit close enough to share a class.
This follows `run_pooled_structural.py`, where the assumption is stated at
length. A null here stays ambiguous between "no effect" and "the classes were
merged wrongly".

Availability is dataset is site. The meta-learner sees a missingness mask, and
that mask *is* the site indicator, so if site predicted group the stack could
launder it into an accuracy. It does not: gong2015 is 24/23 and Julia2018 is
16/16, so each site is internally balanced. The mask-only control below measures
this directly and must land at chance for anything else here to be readable.

Behaviour is near-circular for this contrast -- action gamers outperform
non-gamers on attention tasks, which is why they were recruited -- so brain-only
and behaviour-including families are declared and corrected separately, and the
brain-only number is the one to read.

Usage:
    uv run python scripts/run_stacked_all.py --n-permutations 200
    uv run python scripts/run_stacked_all.py --smoke      # tiny budget, for wiring
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure
from acnets.multiverse import accuracy_interval, fixed_splits
from acnets.multiverse.runner import edge_names, vectorize
from acnets.pipeline import TimeseriesAggregator
from run_brain_behavior import behavioural_performance

GONG = Path('data/gong2015/derivatives')
JULIA = Path('data/Julia2018/derivatives')
JULIA_TS = Path('data/Julia2018')
OUT = Path('outputs/stacked_all')

STRUCT_ATLASES = ('aal', 'harvard-oxford', 'schaefer200')


# --------------------------------------------------------------------------
# data


def load_everything(atlas: str, seed: int):
    """One participant table for the pooled sample, with per-modality masks.

    Returns
    -------
    blocks
        {modality: (n_subjects, n_features) array with NaN rows where absent}
    y
        0 = player (gong expert, Julia AVGP), 1 = non-player.
    site
        0 = gong2015, 1 = Julia2018. Identical to the availability pattern,
        which is exactly why the mask-only control matters.
    """
    gong = pd.read_csv(GONG / f'gmv_gong2015_{atlas}.csv')
    julia = pd.read_csv(JULIA / f'gmv_julia2018_{atlas}.csv')
    drop = ('subject', 'total_gm_mm3', 'brain_mask_cm3')
    shared = [c for c in gong.columns if c in julia.columns and c not in drop]

    subjects = ([f'gong:{s}' for s in gong['subject']]
                + [str(s) for s in julia['subject']])
    y = np.concatenate([
        [0 if str(s).startswith('expert') else 1 for s in gong['subject']],
        [0 if 'AVGP' in str(s) else 1 for s in julia['subject']]])
    site = np.concatenate([np.zeros(len(gong), int), np.ones(len(julia), int)])

    structural = np.vstack([gong[shared].to_numpy(float),
                            julia[shared].to_numpy(float)])

    # functional: 21 network edges, Julia2018 only
    dataset = xr.open_dataset(JULIA_TS / 'timeseries_dosenbach2010.nc5',
                              engine='h5netcdf').load()
    np.random.seed(seed)
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    timeseries = aggregated['timeseries'].transpose(
        'subject', 'timepoint', 'network').values
    nodes = [str(n) for n in aggregated['network'].values]
    measure = ExtraConnectivityMeasure(kind='precision', vectorize=False)
    measure.fit(timeseries)
    connectivity = vectorize(measure.transform(timeseries), False)
    fc_index = {str(s): i for i, s in enumerate(dataset['subject'].values)}
    functional = np.full((len(subjects), connectivity.shape[1]), np.nan)
    for row, name in enumerate(subjects):
        if name in fc_index:
            functional[row] = connectivity[fc_index[name]]

    # behaviour: accuracy, RT, inverse efficiency, Julia2018 and only 25 of them
    performance = behavioural_performance()
    performance.index = [str(i) for i in performance.index]
    behaviour = np.full((len(subjects), performance.shape[1]), np.nan)
    for row, name in enumerate(subjects):
        if name in performance.index:
            behaviour[row] = performance.loc[name].to_numpy(float)

    blocks = {'structural': structural,
              'functional': functional,
              'behaviour': behaviour}
    names = {'structural': shared,
             'functional': edge_names(nodes, False),
             'behaviour': list(performance.columns)}
    return blocks, y, site, subjects, names


def available(block: np.ndarray) -> np.ndarray:
    return ~np.isnan(block).any(axis=1)


# --------------------------------------------------------------------------
# base learners


def harmonise(x_fit, site_fit, x_apply, site_apply):
    """Per-site centring and scaling estimated on training rows only."""
    out = x_apply.copy()
    for value in np.unique(site_apply):
        rows = site_fit == value
        if rows.sum() < 2:
            continue
        centre = x_fit[rows].mean(axis=0)
        spread = x_fit[rows].std(axis=0)
        spread[spread < 1e-12] = 1.0
        out[site_apply == value] = (x_apply[site_apply == value] - centre) / spread
    return out


def base_model(modality: str, seed: int):
    if modality == 'structural':
        return LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())
    return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)


def base_scores(modality: str,
                block: np.ndarray,
                y: np.ndarray,
                site: np.ndarray,
                train: np.ndarray,
                apply_to: np.ndarray,
                seed: int) -> np.ndarray:
    """Decision scores for `apply_to`, from a model fitted on `train`.

    NaN wherever the participant lacks the modality, which the meta-learner
    reads through its mask rather than through an imputed value.
    """
    scores = np.full(len(apply_to), np.nan)
    have = available(block)
    fit_rows = train[have[train]]
    use_rows = apply_to[have[apply_to]]
    if len(fit_rows) < 6 or len(np.unique(y[fit_rows])) < 2 or len(use_rows) == 0:
        return scores

    if modality == 'structural':
        x_fit = harmonise(block[fit_rows], site[fit_rows],
                          block[fit_rows], site[fit_rows])
        x_use = harmonise(block[fit_rows], site[fit_rows],
                          block[use_rows], site[use_rows])
    else:
        scaler = StandardScaler().fit(block[fit_rows])
        x_fit, x_use = scaler.transform(block[fit_rows]), scaler.transform(block[use_rows])

    fitted = base_model(modality, seed).fit(x_fit, y[fit_rows])
    positions = {row: i for i, row in enumerate(apply_to)}
    for row, value in zip(use_rows, fitted.decision_function(x_use), strict=True):
        scores[positions[row]] = value
    return scores


# --------------------------------------------------------------------------
# the stack


def meta_features(scores: dict[str, np.ndarray],
                  modalities: tuple[str, ...]) -> np.ndarray:
    """Scores with missing set to zero, plus one availability indicator each.

    Zero-filling is safe only because the indicator travels with it: the
    meta-learner can tell a genuinely neutral score from an absent one.
    """
    columns = []
    for modality in modalities:
        value = scores[modality]
        mask = ~np.isnan(value)
        columns.append(np.where(mask, np.nan_to_num(value), 0.0))
        if not mask.all():
            columns.append(mask.astype(float))
    return np.column_stack(columns)


def stacked_accuracy(blocks, y, site, modalities, splits, seed, inner_folds=4):
    """Outer-CV accuracy of the stack over every participant."""
    correct, total = 0, 0
    for train, test in splits:
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                random_state=seed)
        train_scores = {m: np.full(len(train), np.nan) for m in modalities}
        for inner_train, inner_test in inner.split(train, y[train]):
            for modality in modalities:
                got = base_scores(modality, blocks[modality], y, site,
                                  train[inner_train], train[inner_test], seed)
                train_scores[modality][inner_test] = got

        test_scores = {m: base_scores(m, blocks[m], y, site, train, test, seed)
                       for m in modalities}

        x_train = meta_features(train_scores, modalities)
        x_test = meta_features(test_scores, modalities)
        if x_train.shape[1] != x_test.shape[1]:
            continue
        meta = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        meta.fit(x_train, y[train])
        correct += int((meta.predict(x_test) == y[test]).sum())
        total += len(test)
    return correct / total if total else float('nan')


def mask_only_accuracy(blocks, y, modalities, splits, seed):
    """The control that has to be at chance: availability alone, no measurements."""
    columns = [available(blocks[m]).astype(float) for m in modalities
               if not available(blocks[m]).all()]
    if not columns:
        return float('nan')
    x = np.column_stack(columns)
    correct, total = 0, 0
    for train, test in splits:
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        model.fit(x[train], y[train])
        correct += int((model.predict(x[test]) == y[test]).sum())
        total += len(test)
    return correct / total if total else float('nan')


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-B', '--n-permutations', type=int, default=200)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        args.n_permutations, args.n_repeats = 3, 2

    families = {
        'brain-only': ('structural', 'functional'),
        'with-behaviour': ('structural', 'functional', 'behaviour'),
    }

    cells, blocks_by_atlas = [], {}
    for atlas in STRUCT_ATLASES:
        blocks, y, site, subjects, names = load_everything(atlas, args.seed)
        blocks_by_atlas[atlas] = (blocks, y, site, subjects, names)
        for family, modalities in families.items():
            cells.append((atlas, family, modalities))

    blocks, y, site, subjects, names = blocks_by_atlas[STRUCT_ATLASES[0]]
    n = len(y)
    splits = fixed_splits(n, args.n_splits, args.n_repeats, random_state=args.seed)
    counts = {m: int(available(blocks[m]).sum()) for m in blocks}
    print(f'pooled n = {n}  ({int((y == 0).sum())} players, '
          f'{int((y == 1).sum())} non-players)')
    print('availability: ' + ', '.join(f'{k} {v}/{n}' for k, v in counts.items()))
    print(f'{len(cells)} cells ({len(STRUCT_ATLASES)} atlases x '
          f'{len(families)} families), {len(splits)} outer folds, '
          f'B = {args.n_permutations}\n')

    control = mask_only_accuracy(blocks, y, families['with-behaviour'],
                                 splits, args.seed)
    print(f'mask-only control (must be ~chance): {control:.3f}\n')

    observed = {}
    for atlas, family, modalities in cells:
        cell_blocks, cell_y, cell_site, *_ = blocks_by_atlas[atlas]
        accuracy = stacked_accuracy(cell_blocks, cell_y, cell_site, modalities,
                                    splits, args.seed)
        observed[f'{atlas}/{family}'] = accuracy
        print(f'  {atlas:15s} {family:15s} {accuracy:.3f}')

    # family-wise null, computed separately for the brain-only and the
    # behaviour-including families: they answer different questions and pooling
    # them would price the honest one against the near-circular one.
    rng = np.random.default_rng(args.seed)
    null = {family: np.empty(args.n_permutations) for family in families}
    started = time.time()
    for i in range(args.n_permutations):
        permuted = rng.permutation(y)
        for family, modalities in families.items():
            best = -np.inf
            for atlas in STRUCT_ATLASES:
                cell_blocks, _, cell_site, *_ = blocks_by_atlas[atlas]
                best = max(best, stacked_accuracy(cell_blocks, permuted,
                                                  cell_site, modalities,
                                                  splits, args.seed))
            null[family][i] = best
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - started) / (i + 1)
            print(f'  {i + 1}/{args.n_permutations} permutations '
                  f'({rate:.1f}s each, '
                  f'~{rate * (args.n_permutations - i - 1) / 60:.1f} min left)')

    report = {'n': n, 'n_players': int((y == 0).sum()),
              'availability': counts, 'mask_only_control': control,
              'n_outer_folds': len(splits), 'n_permutations': args.n_permutations,
              'observed': observed, 'families': {}}

    print()
    for family in families:
        best_cell = max((k for k in observed if k.endswith(family)),
                        key=lambda k: observed[k])
        best = observed[best_cell]
        critical = float(np.percentile(null[family], 95))
        p_value = float((1.0 + np.sum(null[family] >= best))
                        / (args.n_permutations + 1.0))
        interval = accuracy_interval(best * n, n)
        report['families'][family] = {
            'best_cell': best_cell, 'best': best,
            'critical_value_95': critical, 'p_value': p_value,
            'null_median': float(np.median(null[family])),
            'wilson95': [interval.low, interval.high]}
        print(f'{family:15s} best {best:.3f} ({best_cell})  '
              f'critical {critical:.3f}  p = {p_value:.4f}  '
              f'null median {np.median(null[family]):.3f}  '
              f'Wilson [{interval.low:.2f}, {interval.high:.2f}]')

    report['label_assumption'] = (
        "gong2015 non-experts are amateur action gamers, treated here as "
        "non-players. A null is ambiguous between no effect and wrongly merged "
        "classes.")
    report['behaviour_note'] = (
        "the with-behaviour family is near-circular: gamers were recruited for "
        "outperforming non-gamers on attention tasks. Read brain-only.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stacked_all.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {OUT / "stacked_all.json"}')


if __name__ == '__main__':
    main()
