"""Run the cross-specification attribution agreement analysis.

This is the projection that carries the paper's claim that explanation is more
stable across the multiverse than prediction is. It runs the specification grid
on the Julia2018 resting-state data, collects one attribution vector per
specification, and reports agreement between them *against the agreement two
random rankings of the same features would reach*.

The last clause is the whole point. Raw agreement is not interpretable: every
specification ranks the same feature vocabulary, so under a top-weighted measure
two random rankings already agree substantially, and the floor moves with the
number of features. An RBO of 0.85 over 21 network edges is a finding; the same
number over 6 features is close to chance. Every number this script prints is
therefore accompanied by its null.

Scope: agreement needs a shared feature vocabulary, so the primary analysis is
the set of specifications that share one -- the network-scale grid on the
Dosenbach atlas, where every specification ranks the same 21 network edges.
Region-scale and cross-atlas agreement need a crosswalk table mapping regions
between parcellations, which does not exist; those specifications are run and
reported separately, within-atlas only.

Usage:
    uv run python scripts/run_agreement.py                  # the primary analysis
    uv run python scripts/run_agreement.py --include-region # adds region scale
    uv run python scripts/run_agreement.py --n-repeats 5    # quick pass
    uv run python scripts/run_agreement.py --n-jobs 1       # serial, same numbers
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acnets.multiverse import (
    ASYMMETRIC_STATISTICS,
    Specification,
    SpecificationRegistry,
    accuracy_report,
    agreement_matrix,
    agreement_vs_threshold,
    excess_agreement,
    fit_lemon_transforms,
    fixed_splits,
    random_agreement_baseline,
    run_specifications,
    selection_frequency,
    stability_comparison,
    summarize,
    variance_decomposition,
)
from acnets.multiverse.runner import collapse_to_undirected
from acnets.multiverse.specification import SCALE_SENSITIVE_CLASSIFIERS
from acnets.pipeline import TimeseriesAggregator

DATA = Path('data/Julia2018')
LEMON = Path('data/LEMON')
CACHE = Path('outputs/cache/features')
OUT = Path('outputs/agreement')

# Six of the seven statistics. transfer_entropy is excluded because pyinform
# vendors an x86_64-only libinform and dies at call time on arm64 -- see the
# `te` extra in pyproject.toml. Excluded up front rather than left to fail
# mid-grid, since a specification that dies halfway still counts against the
# family-wise correction.
RUNNABLE_STATISTICS = (
    'correlation',
    'partial correlation',
    'covariance',
    'precision',
    'tangent',
    'chatterjee',
)


def load_atlas(atlas: str) -> xr.Dataset:
    path = DATA / f'timeseries_{atlas}.nc5'
    if not path.exists():
        raise SystemExit(
            f'missing {path}. Run `dvc pull data/Julia2018.dvc` first; see '
            f'MERGE_NOTES.md for the state of the DVC pointers.')
    return xr.open_dataset(path, engine='h5netcdf').load()


def group_labels(dataset: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    """AVGP/NVGP from the subject identifier.

    The `group` variable in the netCDF is all-NaN, so the identifier prefix is
    the only surviving record of group membership. This matches how
    Julia2018DataModule derives labels.
    """
    subjects = [str(s) for s in dataset['subject'].values]
    names = sorted({s[:4] for s in subjects})
    if names != ['AVGP', 'NVGP']:
        raise SystemExit(f'unexpected group prefixes: {names}')
    return np.array([0 if s.startswith('AVGP') else 1 for s in subjects]), subjects


def aggregate(dataset: xr.Dataset, seed: int) -> tuple[dict, dict]:
    """Pre-compute each aggregation once; it is per-subject and cannot leak."""
    timeseries, nodes = {}, {}

    region = dataset['timeseries'].transpose('subject', 'timepoint', 'region')
    timeseries['region'] = region.values
    nodes['region'] = [str(r) for r in dataset['region'].values]

    if 'network' in dataset.variables:
        for strategy in ('network', 'random_network'):
            # random_network permutes region-to-network assignment, so it is
            # seeded here rather than left to global numpy state.
            np.random.seed(seed)
            aggregated = TimeseriesAggregator(strategy=strategy).fit_transform(dataset)
            values = aggregated['timeseries'].transpose('subject', 'timepoint', 'network')
            timeseries[strategy] = values.values
            nodes[strategy] = [str(n) for n in aggregated['network'].values]

    return timeseries, nodes


def lemon_transforms(atlas: str, nodes: dict, statistics, window_length: int):
    """LEMON-fitted measures and scalers, or None when LEMON is absent.

    Aggregated to network scale with the same TimeseriesAggregator the target
    data goes through, then windowed to the target's run length. Labels are
    never loaded -- the LEMON metadata is not even in the cached file.
    """
    path = LEMON / f'timeseries_{atlas}.nc'
    if not path.exists():
        print(f'note: {path} missing; lemon-pretrained specifications skipped')
        return None

    dataset = xr.open_dataset(path, engine='h5netcdf').load()
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    values = aggregated['timeseries'].transpose('subject', 'timepoint', 'network').values
    lemon_nodes = [str(n) for n in aggregated['network'].values]
    if lemon_nodes != nodes['network']:
        raise SystemExit(f'LEMON/{atlas} network vocabulary {lemon_nodes} does '
                         f'not match target {nodes["network"]}')

    transforms = fit_lemon_transforms(values, lemon_nodes, statistics,
                                      window_length=window_length)
    print(f'lemon transforms: {transforms.n_windows} windows of '
          f'{window_length} timepoints from {dataset.sizes["subject"]} subjects')
    return transforms


def report_agreement(name: str,
                     attributions: dict,
                     accuracies: dict,
                     n_subjects: int,
                     top_k: int | None,
                     seed: int) -> dict:
    """Agreement, its null, and the excess of one over the other."""
    vocabulary = sorted({f for a in attributions.values() for f in a})

    observed = agreement_matrix(attributions, method='rbo', top_k=top_k,
                                random_state=seed)
    null = random_agreement_baseline(vocabulary, method='rbo', top_k=top_k,
                                     random_state=seed)
    calibrated = excess_agreement(observed.mean, null)

    kendall = agreement_matrix(attributions, method='kendall', random_state=seed)
    kendall_null = random_agreement_baseline(vocabulary, method='kendall',
                                             random_state=seed)
    kendall_calibrated = excess_agreement(kendall.mean, kendall_null)

    thesis = stability_comparison(attributions, accuracies, n_subjects=n_subjects,
                                  method='rbo', top_k=top_k, random_state=seed)

    return {
        'scale': name,
        'n_specifications': observed.n_specifications,
        'n_features': len(vocabulary),
        'rbo': {
            'observed': observed.mean,
            'ci95': [observed.ci_low, observed.ci_high],
            **{k: v for k, v in calibrated.items() if k != 'observed'},
        },
        'kendall': {
            'observed': kendall.mean,
            'ci95': [kendall.ci_low, kendall.ci_high],
            **{k: v for k, v in kendall_calibrated.items() if k != 'observed'},
        },
        'stability_comparison': thesis,
        '_matrix': observed.matrix,
    }


def print_report(report: dict) -> None:
    rbo, ken, thesis = report['rbo'], report['kendall'], report['stability_comparison']
    print(f'\n  {report["scale"]}: '
          f'{report["n_specifications"]} specifications over '
          f'{report["n_features"]} features')
    print(f'    RBO      {rbo["observed"]:.3f} '
          f'[{rbo["ci95"][0]:.3f}, {rbo["ci95"][1]:.3f}]   '
          f'random rankings {rbo["baseline_mean"]:.3f}   '
          f'excess {rbo["normalized"]:+.3f}   p = {rbo["p_value"]:.4f}')
    print(f'    Kendall  {ken["observed"]:.3f} '
          f'[{ken["ci95"][0]:.3f}, {ken["ci95"][1]:.3f}]   '
          f'random rankings {ken["baseline_mean"]:.3f}   '
          f'excess {ken["normalized"]:+.3f}   p = {ken["p_value"]:.4f}')
    print(f'    attribution stability {thesis["attribution"]["normalized"]:+.3f} '
          f'vs accuracy stability {thesis["accuracy"]["normalized"]:+.3f}   '
          f'difference {thesis["difference"]:+.3f}')
    print(f'    explanation more stable than prediction: '
          f'{thesis["thesis_supported"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-repeats', type=int, default=25,
                        help='CV repetitions; 25 x 4 folds = 100 fits per spec')
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=None,
                        help='truncate rankings before comparison')
    parser.add_argument('--include-region', action='store_true',
                        help='also run region-scale specifications (slow)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='worker processes for CV folds; -1 uses every core')
    parser.add_argument('--no-lemon', action='store_true',
                        help='drop the lemon-pretrained specifications')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    dataset = load_atlas('dosenbach2010')
    labels, _subjects = group_labels(dataset)
    n_subjects = len(labels)
    print(f'Julia2018/dosenbach2010: {n_subjects} subjects '
          f'({int((labels == 0).sum())} AVGP, {int((labels == 1).sum())} NVGP), '
          f'{dataset.sizes["timepoint"]} timepoints, '
          f'{dataset.sizes["region"]} regions')

    timeseries, nodes = aggregate(dataset, args.seed)
    print('aggregations: ' + ', '.join(
        f'{k} ({v.shape[-1]} nodes)' for k, v in timeseries.items()))

    aggregations = ['network', 'random_network']
    if args.include_region:
        aggregations.append('region')

    registry = SpecificationRegistry(
        atlases=('dosenbach2010',),
        statistics=RUNNABLE_STATISTICS,
        aggregations=tuple(aggregations),
        # three model classes, so "does the conclusion depend on the model
        # family" is answered rather than assumed: a linear decoder, a tree
        # ensemble, and a small neural net
        classifiers=('svm', 'xgboost', 'mlp'),
        # 'lemon' variants exist at network scale only; the registry's validity
        # rules drop the invalid crossings, so this widens the grid by exactly
        # the runnable transfer specifications.
        pretrainings=('none',) if args.no_lemon else ('none', 'lemon'))
    specs = registry.specifications

    lemon = None
    if any(s.pretraining == 'lemon' for s in specs):
        lemon = lemon_transforms('dosenbach2010', nodes, RUNNABLE_STATISTICS,
                                 window_length=dataset.sizes['timepoint'])
        if lemon is None:  # LEMON data absent: shrink the grid, say so
            specs = [s for s in specs if s.pretraining == 'none']

    n_jobs = args.n_jobs if args.n_jobs != -1 else (os.cpu_count() or 1)
    print(f'\nrunning {len(specs)} specifications '
          f'({args.n_splits} folds x {args.n_repeats} repeats each) '
          f'on {n_jobs} workers\n')

    started = time.perf_counter()

    def progress(spec: Specification) -> None:
        print(f'  {spec.label}', flush=True)

    # One label-independent partition, shared with run_permutation.py: the
    # observed accuracies and the null are then computed on identical folds,
    # and the feature cache built here is reused verbatim by every permutation.
    splits = fixed_splits(n_subjects, args.n_splits, args.n_repeats, args.seed)

    outcome = run_specifications(
        specs, timeseries, nodes, labels,
        progress=progress,
        n_splits=args.n_splits, n_repeats=args.n_repeats,
        random_state=args.seed, n_jobs=args.n_jobs,
        lemon=lemon, cache_dir=CACHE, splits=splits)

    elapsed = time.perf_counter() - started
    print(f'\n{len(specs) * args.n_splits * args.n_repeats} fits in '
          f'{elapsed:.1f}s')

    # ---- per-specification table -------------------------------------------
    rows = []
    for result in outcome.results.values():
        row = result.summary()
        if result.failed is None:
            report = accuracy_report(result.fold_accuracies, n_subjects)
            row.update({
                'split_stability_sd': report.estimator_sd,
                'sampling_ci_low': report.sampling_low,
                'sampling_ci_high': report.sampling_high,
            })
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    table.to_csv(OUT / 'specifications.csv', index=False)

    print(f'\naccuracy across {len(table)} specifications: '
          f'{table["accuracy"].min():.3f} to {table["accuracy"].max():.3f} '
          f'(median {table["accuracy"].median():.3f})')

    # ---- which choices matter (manuscript 3.4.2) ---------------------------
    # Two decompositions, because the grid has structural confounds that make a
    # single decomposition meaningless. Lemon pretraining exists at network
    # scale only, so pretraining and aggregation cannot be separated over the
    # full grid: the first decomposition drops pretraining and uses the 'none'
    # subset (statistic x aggregation x classifier, fully crossed).
    #
    # And within network scale, pretraining is a no-op for a scale-invariant
    # classifier on a stateless statistic, so those cells do not exist
    # (Specification.is_valid). Crossing pretraining with all three classifiers
    # would therefore be unbalanced by construction. The second decomposition
    # restricts to the scale-sensitive classifiers, where every statistic
    # realises both pretraining levels: a fully crossed
    # statistic x classifier x pretraining design.
    def decompose(name, subset):
        keep = {s.spec_id for s in subset}
        accuracies = {k: v for k, v in outcome.accuracies.items() if k in keep}
        frame = variance_decomposition(accuracies, subset)
        print(f'\nvariance in accuracy, {name} '
              f'(R2_full = {frame.attrs["r2_full"]:.2f}):')
        for dimension, partial in frame['partial'].sort_values(ascending=False).items():
            levels = int(frame.loc[dimension, 'n_levels'])
            if levels < 2:
                continue
            print(f'  {dimension:12s} partial {partial:.3f}  '
                  f'marginal {frame.loc[dimension, "marginal"]:.3f}  '
                  f'({levels} levels)')
        return summarize(frame)

    variance_payload = {
        'no_pretraining_subset': decompose(
            'no-pretraining subset',
            [s for s in specs if s.pretraining == 'none']),
    }
    if any(s.pretraining == 'lemon' for s in specs):
        variance_payload['pretraining_subset'] = decompose(
            'network scale, scale-sensitive classifiers (pretraining crossed)',
            [s for s in specs if s.aggregation == 'network'
             and s.classifier in SCALE_SENSITIVE_CLASSIFIERS])
    (OUT / 'variance.json').write_text(
        json.dumps(variance_payload, indent=2, default=float))
    print('\ntop 5 by accuracy:')
    for _, r in table.head(5).iterrows():
        print(f'  {r["accuracy"]:.3f}  [{r["sampling_ci_low"]:.2f}, '
              f'{r["sampling_ci_high"]:.2f}]  {r["label"]}')

    # ---- agreement, by scale, within a shared vocabulary --------------------
    # Directed statistics are collapsed onto the undirected edge vocabulary
    # first. Without it the directed specifications share no feature names with
    # the symmetric ones and their agreement is exactly zero by construction,
    # which would read as disagreement rather than as the naming artefact it is.
    reports = []
    for scale in aggregations:
        subset = {r.spec.label: r for r in outcome.results.values()
                  if r.failed is None and r.spec.aggregation == scale}
        if len(subset) < 2:
            continue
        report = report_agreement(
            scale,
            {k: collapse_to_undirected(r.attribution) for k, r in subset.items()},
            {k: r.accuracy for k, r in subset.items()},
            n_subjects, args.top_k, args.seed)
        matrix = report.pop('_matrix')
        matrix.to_csv(OUT / f'agreement_matrix_{scale}.csv')
        reports.append(report)
        print_report(report)

        # Sensitivity check: the same statistic restricted to the specifications
        # that never needed collapsing. If the collapse were manufacturing the
        # result, this number would be markedly lower.
        symmetric = {k: r for k, r in subset.items()
                     if r.spec.statistic not in ASYMMETRIC_STATISTICS}
        if len(symmetric) >= 2:
            check = report_agreement(
                f'{scale} (symmetric statistics only)',
                {k: r.attribution for k, r in symmetric.items()},
                {k: r.accuracy for k, r in symmetric.items()},
                n_subjects, args.top_k, args.seed)
            check.pop('_matrix')
            reports.append(check)
            print_report(check)

    # ---- which features are reproducibly prominent -------------------------
    network_specs = {r.spec.label: r for r in outcome.results.values()
                     if r.failed is None and r.spec.aggregation == 'network'}
    frequency = selection_frequency(
        {k: collapse_to_undirected(r.attribution)
         for k, r in network_specs.items()}, top_k=5)
    frequency.to_csv(OUT / 'selection_frequency_network.csv')
    print('\nselection frequency, top-5 across network specifications:')
    for name, value in frequency.head(8).items():
        print(f'  {value:.2f}  {name}')

    # ---- does conditioning on accuracy manufacture the agreement? -----------
    curve = agreement_vs_threshold(
        {k: collapse_to_undirected(r.attribution)
         for k, r in network_specs.items()},
        {k: r.accuracy for k, r in network_specs.items()},
        method='rbo', top_k=args.top_k, random_state=args.seed)
    curve.to_csv(OUT / 'agreement_vs_threshold.csv', index=False)
    print('\nagreement vs accuracy cutoff (network scale):')
    for _, r in curve.iterrows():
        n = int(r['n_specifications'])
        value = ('    n/a' if np.isnan(r['agreement'])
                 else f'{r["agreement"]:.3f}')
        print(f'  >= {r["threshold"]:.2f}   n = {n:2d}   {value}')

    (OUT / 'agreement.json').write_text(json.dumps({
        'dataset': 'Julia2018',
        'atlas': 'dosenbach2010',
        'n_subjects': n_subjects,
        'n_splits': args.n_splits,
        'n_repeats': args.n_repeats,
        'statistics': list(RUNNABLE_STATISTICS),
        'excluded_statistics': ['transfer_entropy'],
        'reports': reports,
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
