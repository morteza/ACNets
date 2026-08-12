"""Multi-atlas learned parcellation: nine atlases, one shared network space.

Extends `run_learned_parcellation.py` from one atlas to several. Each atlas
learns its own region-to-network assignment, initialised from the hand-declared
crosswalk in `acnets.deep.canonical_networks`, and all of them pool into the
same seven canonical networks -- so the middle layer, the attention over
atlases, and the head are shared, and the attribution comes out in one
vocabulary rather than nine.

Four numbers are reported and every one is priced against permuted labels:

  accuracy      does it decode
  attribution   are the canonical network edges it leans on label-dependent
  drift         does each atlas's learned parcellation leave the crosswalk,
                and does it leave by the same distance when labels are shuffled
  atlas weights which parcellations the model actually uses

Drift is the diagnostic that matters. About 9,500 assignment parameters face 24
training participants, so the assignments *can* move anywhere; the question is
whether they move somewhere the labels chose. Equal drift on real and shuffled
labels means they did not.

    uv run python scripts/run_learned_parcellation_multi.py
    uv run python scripts/run_learned_parcellation_multi.py --random-init
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn

from acnets.connectome import ExtraConnectivityMeasure
from acnets.deep.canonical_networks import CANONICAL, assignment_matrix
from acnets.deep.region_to_network import MultiAtlasRegionToNetwork
from acnets.multiverse import accuracy_interval, fixed_splits

warnings.filterwarnings('ignore')

TIMESERIES = Path('data/Julia2018/derivatives/resting_timeseries')
OUT = Path('outputs/learned_parcellation')

# The seven atlases whose canonical networks intersect in a usable space.
# msdl and dosenbach2007 are deliberately not here: dosenbach2007 defines only
# CON, FPN and CER, and msdl has no FPN, so including either collapses the
# common vocabulary to CON alone. Both remain available through --atlases, and
# --all-networks keeps every network at the cost described in
# `restrict_to_common`.
ATLASES = ('dosenbach2010', 'gordon2014_2mm', 'seitzman2018', 'schaefer200-7',
           'schaefer100-7', 'difumo_64_2mm', 'difumo_128_2mm')


def restrict_to_common(assignments, used, names):
    """Keep only canonical networks every included atlas can measure.

    A shared vocabulary makes atlases comparable; it does not make them equally
    populated. Cerebellum exists in two of these atlases and dorsal attention in
    all but one, so an edge touching cerebellum is estimated from a fifth of the
    data an edge touching default mode is. Neither pooling scheme fixes that:
    weighting sparse edges equally under-counts them, and renormalising by the
    weight available amplifies them, because dividing a mean by a small
    denominator inflates its variance. Both were tried and both distorted the
    attribution -- towards the dense edges in the first case, towards the sparse
    ones in the second.

    Restricting to the intersection is the honest resolution. It costs the
    networks that only some atlases define and buys an attribution whose edges
    are all estimated from the same number of atlases, which is the condition
    under which comparing them means anything.
    """
    present = None
    for matrix in assignments:
        available = (matrix.sum(dim=0) > 0)
        present = available if present is None else (present & available)
    keep = present.nonzero(as_tuple=True)[0].tolist()
    kept_names = [names[i] for i in keep]
    dropped = [n for i, n in enumerate(names) if i not in keep]
    restricted = [matrix[:, keep] for matrix in assignments]
    print(f'  common networks: {", ".join(kept_names)}'
          + (f'   (dropped {", ".join(dropped)})' if dropped else ''))
    return restricted, kept_names


def load(atlases, statistic):
    series, assignments, used, labels = [], [], [], None
    for atlas in atlases:
        path = TIMESERIES / f'timeseries_{atlas}.nc5'
        if not path.exists():
            print(f'  note: {atlas} missing, skipped')
            continue
        dataset = xr.open_dataset(path, engine='h5netcdf').load()
        matrix, report = assignment_matrix(atlas, dataset['network'].values)
        timeseries = dataset['timeseries'].transpose(
            'subject', 'timepoint', 'region').values

        subjects = [str(s) for s in dataset['subject'].values]
        if labels is None:
            labels = np.array([0 if 'AVGP' in s else 1 for s in subjects])
        series.append(timeseries)
        assignments.append(torch.tensor(matrix, dtype=torch.float32))
        used.append(atlas)
        print(f'  {atlas:16s} {report["n_regions"]:4d} regions   '
              f'crosswalk places {report["coverage"]:.0%}')
    return series, assignments, used, labels


def fold_connectivity(series, statistic, train_idx, test_idx):
    trains, tests = [], []
    for timeseries in series:
        measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
        measure.fit(timeseries[train_idx])
        trains.append(torch.tensor(
            np.asarray(measure.transform(timeseries[train_idx]), dtype='float32')))
        tests.append(torch.tensor(
            np.asarray(measure.transform(timeseries[test_idx]), dtype='float32')))
    return trains, tests


def fit(model, x, y, epochs, lr, weight_decay):
    """Assignments exempt from weight decay -- see run_learned_parcellation.py."""
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        (undecayed if 'assignment_logits' in name else decayed).append(parameter)
    optimiser = torch.optim.Adam(
        [{'params': decayed, 'weight_decay': weight_decay},
         {'params': undecayed, 'weight_decay': 0.0}], lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        model.anneal(epoch / epochs)
        optimiser.zero_grad()
        criterion(model(x), y).backward()
        optimiser.step()
    model.eval()


def evaluate(folds, labels, assignments, region_counts, args):
    correct = total = 0
    train_accuracies, attributions, drifts, weights = [], [], [], []
    init = None if args.random_init else assignments

    for fold in folds:
        train_idx, test_idx = fold['train_idx'], fold['test_idx']
        y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
        torch.manual_seed(args.seed)
        model = MultiAtlasRegionToNetwork(
            region_counts=region_counts, n_networks=args.n_networks_resolved,
            hidden=tuple(args.hidden), dropout=args.dropout,
            init_assignments=init)
        fit(model, fold['train'], y_train, args.epochs, args.lr,
            args.weight_decay)

        with torch.no_grad():
            predicted = (model(fold['test']).numpy() > 0).astype(int)
            train_predicted = (model(fold['train']).numpy() > 0).astype(int)
        correct += int((predicted == labels[test_idx]).sum())
        total += len(test_idx)
        train_accuracies.append(float((train_predicted == labels[train_idx]).mean()))
        attributions.append(model.edge_attribution(fold['train']).numpy())
        drifts.append(model.assignment_drift())
        weights.append(model.atlas_weights().numpy())

    return {'accuracy': correct / total,
            'train_accuracy': float(np.mean(train_accuracies)),
            'attribution': np.mean(attributions, axis=0),
            'drift': np.nanmean(np.array(drifts, dtype=float), axis=0),
            'atlas_weights': np.mean(weights, axis=0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlases', nargs='+', default=list(ATLASES))
    parser.add_argument('--statistic', default='partial correlation')
    parser.add_argument('-B', '--n-permutations', type=int, default=100)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=1)
    parser.add_argument('--hidden', type=int, nargs='+', default=[16, 8])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--all-networks', action='store_true',
                        help='keep canonical networks even when only some '
                             'atlases define them; the attribution then mixes '
                             'edges estimated from different numbers of atlases')
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print('atlases:')
    series, assignments, used, labels = load(args.atlases, args.statistic)
    network_names = list(CANONICAL)
    if not args.all_networks:
        assignments, network_names = restrict_to_common(assignments, used,
                                                        network_names)
    region_counts = [s.shape[2] for s in series]
    n = len(labels)
    splits = fixed_splits(n, args.n_splits, args.n_repeats,
                          random_state=args.seed)

    probe = MultiAtlasRegionToNetwork(region_counts, len(network_names),
                                      tuple(args.hidden))
    print(f'\n{len(used)} atlases, {sum(region_counts)} regions total, '
          f'{n} participants, {len(splits)} folds')
    print(f'{sum(p.numel() for p in probe.parameters())} parameters '
          f'({sum(region_counts) * len(network_names)} of them assignments) against '
          f'{int(n * (args.n_splits - 1) / args.n_splits)} training participants')
    print(f'{args.statistic}, {"random" if args.random_init else "crosswalk"} '
          f'init, B = {args.n_permutations}\n')

    args.n_networks_resolved = len(network_names)
    folds = []
    for train_idx, test_idx in splits:
        train, test = fold_connectivity(series, args.statistic, train_idx,
                                        test_idx)
        folds.append({'train_idx': train_idx, 'test_idx': test_idx,
                      'train': train, 'test': test})

    started = time.time()
    observed = evaluate(folds, labels, assignments, region_counts, args)
    per_run = time.time() - started
    interval = accuracy_interval(observed['accuracy'] * n, n)
    print(f'observed   test {observed["accuracy"]:.4f}   '
          f'train {observed["train_accuracy"]:.4f}   ({per_run:.0f}s per run)')
    print(f'           Wilson 95%: [{interval.low:.2f}, {interval.high:.2f}]')
    print(f'estimated total: {per_run * (args.n_permutations + 1) / 60:.1f} min')

    rows, cols = np.triu_indices(len(network_names))
    edge_labels = [f'{network_names[i]}-{network_names[j]}'
                   for i, j in zip(rows, cols, strict=True)]
    print('\ntop canonical network edges (shared across all atlases):')
    for i in np.argsort(-observed['attribution'])[:6]:
        print(f'  {edge_labels[i]:12s} {observed["attribution"][i]:.4f}')

    print('\natlas weights and learned-parcellation drift:')
    for i in np.argsort(-observed['atlas_weights']):
        print(f'  {used[i]:16s} weight {observed["atlas_weights"][i]:.4f}   '
              f'drift {observed["drift"][i]:.3f}')
    print(f'  uniform weight would be {1 / len(used):.4f}')

    rng = np.random.default_rng(args.seed)
    null_accuracy = np.empty(args.n_permutations)
    null_drift = np.empty((args.n_permutations, len(used)))
    null_agreement = np.empty(args.n_permutations)
    started = time.time()
    for i in range(args.n_permutations):
        result = evaluate(folds, rng.permutation(labels), assignments,
                          region_counts, args)
        null_accuracy[i] = result['accuracy']
        null_drift[i] = result['drift']
        null_agreement[i] = float(np.corrcoef(
            np.argsort(np.argsort(-result['attribution'])),
            np.argsort(np.argsort(-observed['attribution'])))[0, 1])
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - started) / (i + 1)
            print(f'  {i + 1}/{args.n_permutations} ({rate:.1f}s each, '
                  f'~{rate * (args.n_permutations - i - 1) / 60:.1f} min left)')

    p_accuracy = float((1.0 + np.sum(null_accuracy >= observed['accuracy']))
                       / (args.n_permutations + 1.0))
    print(f'\naccuracy     observed {observed["accuracy"]:.4f}   '
          f'null median {np.median(null_accuracy):.4f}   '
          f'95th {np.percentile(null_accuracy, 95):.4f}   p = {p_accuracy:.4f}')
    print(f'attribution  rank correlation with permuted-label runs: '
          f'{null_agreement.mean():+.3f} '
          f'[{np.percentile(null_agreement, 2.5):+.3f}, '
          f'{np.percentile(null_agreement, 97.5):+.3f}]')
    print('\ndrift, observed against permuted-label median:')
    for i, atlas in enumerate(used):
        print(f'  {atlas:16s} {observed["drift"][i]:.3f} vs '
              f'{np.median(null_drift[:, i]):.3f}')
    print('  a gap here would mean the labels chose where the parcellation '
          'moved; no gap means it moved regardless')

    report = {
        'atlases': used, 'networks': network_names,
        'statistic': args.statistic, 'n': int(n),
        'n_folds': len(splits), 'random_init': args.random_init,
        'n_parameters': int(sum(p.numel() for p in probe.parameters())),
        'observed_accuracy': observed['accuracy'],
        'train_accuracy': observed['train_accuracy'],
        'wilson95': [interval.low, interval.high],
        'accuracy_null': {'median': float(np.median(null_accuracy)),
                          'q95': float(np.percentile(null_accuracy, 95)),
                          'p_value': p_accuracy},
        'attribution': dict(zip(edge_labels, observed['attribution'].tolist(),
                                strict=True)),
        'attribution_permuted_rank_correlation': {
            'mean': float(null_agreement.mean()),
            'ci95': [float(np.percentile(null_agreement, 2.5)),
                     float(np.percentile(null_agreement, 97.5))]},
        'atlas_weights': dict(zip(used, observed['atlas_weights'].tolist(),
                                  strict=True)),
        'drift': dict(zip(used, observed['drift'].tolist(), strict=True)),
        'drift_null_median': dict(zip(used, np.median(null_drift, axis=0).tolist(),
                                      strict=True)),
        'n_permutations': args.n_permutations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '_random_init' if args.random_init else ''
    (OUT / f'multi_atlas{suffix}.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {OUT / f"multi_atlas{suffix}.json"}')


if __name__ == '__main__':
    main()
