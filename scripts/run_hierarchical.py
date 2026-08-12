"""Fit the hierarchical multi-atlas connectivity model, and price it honestly.

Builds one input per (atlas, connectivity statistic), maps each atlas's regions
onto a shared seven-network vocabulary with a fixed hand-declared crosswalk, and
fits a single model with an encoder shared across all views
(`acnets.deep.hierarchical`). Nine atlases by six statistics is up to 54 views
over the same 32 participants.

**What this is for.** Not accuracy. Section 3.7 puts the reliability ceiling at
0.72 against a family-wise threshold of 0.739, and section 3.8 records that
thirteen classifier families, nested tuning, external pretraining, modality
fusion, instance-level learning and diffusion normative modelling all fail to
move the result. There is no reason to expect an architecture to, and the null
below is set up to say so cleanly if it does not.

What the model does buy is a feature space that does not exist anywhere else in
this project: **canonical network edges that mean the same thing in every
atlas**. The manuscript's attribution-agreement analysis is confined to one atlas
because agreement needs a shared vocabulary, and section 3.4.3 has just shown
that the permuted-label null is cleared only by a grid that varies the model
family -- diversity, not size, is what makes the null informative. The atlas axis
is the largest unexploited source of that diversity: ten declared, one run. This
script is the instrument for adding it.

**Everything is priced.** Accuracy gets a permuted-label null. So does the
attribution, over the canonical edges, because after section 3.4.3 no attribution
in this project is reported without one. Folds are label-independent and fixed,
and connectivity is estimated inside training folds so the reference-based
statistics cannot leak.

    uv run python scripts/run_hierarchical.py --smoke
    uv run python scripts/run_hierarchical.py -B 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn

from acnets.connectome import ExtraConnectivityMeasure
from acnets.deep.canonical_networks import (
    CANONICAL,
    assignment_matrix,
    edge_names,
    pool_connectivity,
)
from acnets.deep.hierarchical import HierarchicalConnectivityModel
from acnets.deep.network_gnn import NetworkGraphClassifier
from acnets.multiverse import accuracy_interval, fixed_splits

TIMESERIES = Path('data/Julia2018/derivatives/resting_timeseries')
OUT = Path('outputs/hierarchical')

ATLASES = ('dosenbach2010', 'gordon2014_2mm', 'seitzman2018', 'schaefer200-7',
           'schaefer100-7', 'difumo_64_2mm', 'difumo_128_2mm', 'msdl',
           'dosenbach2007')
# chatterjee is excluded by default: it is O(R^2) rank computations per subject
# and Gordon 2014 has 333 regions, which costs more than the rest of the grid
# combined. --with-chatterjee puts it back.
STATISTICS = ('correlation', 'partial correlation', 'covariance', 'precision',
              'tangent')


def load_views(atlases, statistics, strict: bool):
    """Per-view region timeseries plus the fixed region-to-canonical matrix."""
    views, reports = [], []
    labels = None
    for atlas in atlases:
        path = TIMESERIES / f'timeseries_{atlas}.nc5'
        if not path.exists():
            print(f'  note: {path.name} missing, skipped')
            continue
        dataset = xr.open_dataset(path, engine='h5netcdf').load()
        if 'network' not in dataset.variables and 'network' not in dataset.coords:
            print(f'  note: {atlas} has no network labels, skipped')
            continue

        matrix, report = assignment_matrix(atlas, dataset['network'].values,
                                           strict=strict)
        reports.append(report)
        timeseries = dataset['timeseries'].transpose(
            'subject', 'timepoint', 'region').values

        subjects = [str(s) for s in dataset['subject'].values]
        if labels is None:
            labels = np.array([0 if 'AVGP' in s else 1 for s in subjects])
            order = subjects
        elif subjects != order:
            raise SystemExit(f'{atlas} subject order differs from the first atlas')

        node_mask = matrix.sum(axis=0) > 0
        for statistic in statistics:
            views.append({'atlas': atlas, 'statistic': statistic,
                          'timeseries': timeseries, 'matrix': matrix,
                          'node_mask': node_mask})
        print(f'  {atlas:16s} {report["n_regions"]:4d} regions  '
              f'coverage {report["coverage"]:.2f}  '
              f'networks {",".join(report["present"])}')
    return views, labels, reports


def view_features(view, train_idx, test_idx):
    """Canonical connectivity for train and test, from ONE fit per view per fold.

    The measure is fitted on the training fold and applied to both, which is the
    same discipline as `acnets.multiverse.runner._fold_features`. Fitting once
    and transforming twice rather than refitting for each half matters here:
    Gordon 2014 has 333 regions and the covariance estimate dominates the whole
    precompute.
    """
    measure = ExtraConnectivityMeasure(kind=view['statistic'], vectorize=False)
    measure.fit(view['timeseries'][train_idx])
    train = pool_connectivity(
        np.asarray(measure.transform(view['timeseries'][train_idx])),
        view['matrix'])
    test = pool_connectivity(
        np.asarray(measure.transform(view['timeseries'][test_idx])),
        view['matrix'])
    return train[0], test[0], train[1]


def build_fold(views, train_idx, test_idx):
    """(n_train, V, 28), (n_test, V, 28) features and the (V, 28) edge mask."""
    trains, tests, masks = [], [], []
    for view in views:
        train, test, mask = view_features(view, train_idx, test_idx)
        trains.append(train)
        tests.append(test)
        masks.append(mask)
    node_masks = np.stack([v['node_mask'] for v in views], axis=0)
    return (np.stack(trains, axis=1), np.stack(tests, axis=1),
            np.stack(masks, axis=0), node_masks)


def standardise(train, apply_to, edge_mask):
    """Per-view per-edge z-scoring on training statistics only."""
    centre = train.mean(axis=0, keepdims=True)
    spread = train.std(axis=0, keepdims=True)
    spread[spread < 1e-9] = 1.0
    out = (apply_to - centre) / spread
    return out * edge_mask[None]


def make_model(args, n_views, n_edges, seed):
    torch.manual_seed(seed)
    if args.model == 'gnn':
        return NetworkGraphClassifier(
            n_views=n_views, n_nodes=len(CANONICAL), node_dim=args.node_dim,
            n_layers=args.n_layers, readout=args.readout,
            view_attention=args.view_attention, dropout=args.dropout)
    return HierarchicalConnectivityModel(
        n_views=n_views, n_edges=n_edges, embedding_dim=args.embedding_dim,
        view_attention=args.view_attention, dropout=args.dropout)


def fit_fold(fold, y_train, args, seed):
    """Train end to end on one fold; return the model and both accuracies.

    Training accuracy is returned alongside test accuracy because the gap is
    the diagnostic that matters at this sample size. A 75-parameter model on
    24 participants can memorise the training fold and predict the test fold
    anti-correlated with truth, which reads as a below-chance result and is a
    fact about optimisation rather than about the data.
    """
    x_train, x_test = fold['x_train'], fold['x_test']
    model = make_model(args, x_train.shape[1], x_train.shape[2], seed)
    mask = torch.tensor(
        fold['node_mask'] if args.model == 'gnn' else fold['edge_mask'],
        dtype=torch.bool)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(args.epochs):
        optimiser.zero_grad()
        criterion(model(xt, mask), yt).backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(xt, mask)
        test_logits = model(torch.tensor(x_test, dtype=torch.float32), mask)
    train_accuracy = float(((train_logits.numpy() > 0).astype(int) == y_train).mean())
    return model, (test_logits.numpy() > 0).astype(int), train_accuracy, mask


def precompute(views, splits):
    """Standardised features for every fold, computed once.

    Nothing here sees a label -- connectivity is estimated on the training fold
    and the standardisation uses training statistics only -- so the same tensors
    are valid for the observed run and for every permutation. This is the same
    argument that makes the feature cache valid in `acnets.multiverse.runner`,
    and it is what turns the permutation null from hours into minutes: only the
    84-parameter model is refitted per permutation.
    """
    folds = []
    for i, (train_idx, test_idx) in enumerate(splits):
        raw_train, raw_test, edge_mask, node_mask = build_fold(
            views, train_idx, test_idx)
        folds.append({
            'train_idx': train_idx, 'test_idx': test_idx,
            'edge_mask': edge_mask, 'node_mask': node_mask,
            'x_train': standardise(raw_train, raw_train, edge_mask),
            'x_test': standardise(raw_train, raw_test, edge_mask)})
        print(f'  fold {i + 1}/{len(splits)}', end='\r', flush=True)
    print(' ' * 40, end='\r')
    return folds


def evaluate(folds, labels, args, seed, want_attribution=False):
    """Accuracy over the precomputed folds, optionally with attribution."""
    correct = total = 0
    attributions, view_weights = [], []
    train_accuracies, node_importances = [], []
    for fold in folds:
        train_idx, test_idx = fold['train_idx'], fold['test_idx']
        model, predicted, train_accuracy, mask = fit_fold(
            fold, labels[train_idx], args, seed)
        correct += int((predicted == labels[test_idx]).sum())
        total += len(test_idx)
        train_accuracies.append(train_accuracy)

        if want_attribution:
            xt = torch.tensor(fold['x_train'], dtype=torch.float32)
            attributions.append(model.attribution(xt, mask).numpy())
            weights = model.reported_view_weights()
            if weights is not None:
                view_weights.append(weights.numpy())
            if args.model == 'gnn':
                node_importances.append(model.node_importance(xt, mask).numpy())

    accuracy = correct / total if total else float('nan')
    summary = {'train_accuracy': float(np.mean(train_accuracies))}
    if not want_attribution:
        return accuracy, None, None, summary
    summary['node_importance'] = (np.mean(node_importances, axis=0)
                                  if node_importances else None)
    return (accuracy,
            np.mean(attributions, axis=0),
            np.mean(view_weights, axis=0) if view_weights else None,
            summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='gnn', choices=('gnn', 'shared'),
                        help="'gnn' is the end-to-end graph network whose nodes "
                             "are canonical networks carrying vectors; 'shared' "
                             "is the earlier flat encoder over network edges")
    parser.add_argument('-B', '--n-permutations', type=int, default=100)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=2)
    parser.add_argument('--node-dim', type=int, default=4,
                        help='width of each canonical network node; 4-8 is the '
                             'range worth searching at n = 32')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--readout', default='attention',
                        choices=('attention', 'mean'))
    parser.add_argument('--embedding-dim', type=int, default=1)
    parser.add_argument('--view-attention', default='learned',
                        choices=('learned', 'content'))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--with-chatterjee', action='store_true',
                        help='add the rank-based statistic; slow on the large '
                             'atlases')
    parser.add_argument('--strict-crosswalk', action='store_true',
                        help='drop the ventral-attention family instead of '
                             'merging it into CON')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        args.n_permutations, args.n_repeats, args.epochs = 3, 1, 40

    print('loading views:')
    statistics = STATISTICS + (('chatterjee',) if args.with_chatterjee else ())
    views, labels, reports = load_views(ATLASES, statistics,
                                        args.strict_crosswalk)
    if not views:
        raise SystemExit('no views loaded')
    n = len(labels)
    splits = fixed_splits(n, args.n_splits, args.n_repeats,
                          random_state=args.seed)
    print(f'\n{len(views)} views '
          f'({len(views) // len(statistics)} atlases x {len(statistics)} statistics), '
          f'{n} participants, {len(splits)} folds')

    probe = make_model(args, len(views), len(edge_names()), args.seed)
    n_parameters = sum(p.numel() for p in probe.parameters())
    print(f'model: {args.model}'
          + (f', node_dim={args.node_dim}, layers={args.n_layers}, '
             f'readout={args.readout}' if args.model == 'gnn' else ''))
    print(f'{n_parameters} parameters against '
          f'{int(n * (args.n_splits - 1) / args.n_splits)} training participants '
          f'per fold\n')

    print('precomputing fold features (label-independent, reused by every '
          'permutation):')
    folds = precompute(views, splits)

    started = time.time()
    observed, attribution, weights, summary = evaluate(
        folds, labels, args, args.seed, want_attribution=True)
    print(f'observed accuracy: {observed:.4f}   '
          f'(train {summary["train_accuracy"]:.4f}; '
          f'{time.time() - started:.0f}s per evaluation)')
    if summary['train_accuracy'] - observed > 0.35:
        print('  NOTE: train-test gap above 0.35 -- the model is memorising the '
              'training fold; reduce --epochs or raise --weight-decay before '
              'reading anything else here')
    interval = accuracy_interval(observed * n, n)
    print(f'  Wilson 95% at n = {n}: [{interval.low:.2f}, {interval.high:.2f}]')

    names = edge_names()
    ranked = np.argsort(-attribution)
    print('\ntop canonical network edges (mean |grad x input|, shared vocabulary):')
    for i in ranked[:8]:
        print(f'  {names[i]:16s} {attribution[i]:.4f}')

    if summary.get('node_importance') is not None:
        print('\nnode importance (gradient norm at each canonical network):')
        for i in np.argsort(-summary['node_importance']):
            print(f'  {CANONICAL[i]:6s} {summary["node_importance"][i]:.4f}')

    if weights is not None:
        print('\nview weights (softmax over atlas x statistic), top 8:')
        for i in np.argsort(-weights)[:8]:
            print(f'  {views[i]["atlas"]:16s} {views[i]["statistic"]:20s} '
                  f'{weights[i]:.4f}')
        print(f'  uniform would be {1 / len(views):.4f}')

    rng = np.random.default_rng(args.seed)
    null_accuracy = np.empty(args.n_permutations)
    null_agreement = np.empty(args.n_permutations)
    started = time.time()
    for i in range(args.n_permutations):
        permuted = rng.permutation(labels)
        accuracy, permuted_attribution, _, _ = evaluate(
            folds, permuted, args, args.seed, want_attribution=True)
        null_accuracy[i] = accuracy
        # rank correlation between the permuted run's attribution and the
        # observed one: how much of the observed ranking survives label
        # destruction (section 3.4.3's question, on this feature space)
        null_agreement[i] = float(np.corrcoef(
            np.argsort(np.argsort(-permuted_attribution)),
            np.argsort(np.argsort(-attribution)))[0, 1])
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - started) / (i + 1)
            print(f'  {i + 1}/{args.n_permutations} '
                  f'({rate:.1f}s each, '
                  f'~{rate * (args.n_permutations - i - 1) / 60:.1f} min left)')

    p_accuracy = float((1.0 + np.sum(null_accuracy >= observed))
                       / (args.n_permutations + 1.0))
    print(f'\naccuracy    observed {observed:.4f}  '
          f'null median {np.median(null_accuracy):.4f}  '
          f'95th {np.percentile(null_accuracy, 95):.4f}  '
          f'p = {p_accuracy:.4f}')
    print(f'attribution rank correlation with permuted-label runs: '
          f'mean {null_agreement.mean():+.3f} '
          f'[{np.percentile(null_agreement, 2.5):+.3f}, '
          f'{np.percentile(null_agreement, 97.5):+.3f}]')
    print('  near zero means the observed ranking is label-dependent; '
          'near one means it is not')

    report = {
        'n_views': len(views), 'n_participants': int(n),
        'n_parameters': int(n_parameters),
        'embedding_dim': args.embedding_dim,
        'view_attention': args.view_attention,
        'strict_crosswalk': args.strict_crosswalk,
        'crosswalk_coverage': {r['atlas']: r['coverage'] for r in reports},
        'model': args.model,
        'node_dim': args.node_dim if args.model == 'gnn' else None,
        'observed_accuracy': observed,
        'train_accuracy': summary['train_accuracy'],
        'node_importance': (dict(zip(CANONICAL,
                                     summary['node_importance'].tolist(),
                                     strict=True))
                            if summary.get('node_importance') is not None
                            else None),
        'wilson95': [interval.low, interval.high],
        'accuracy_null': {'median': float(np.median(null_accuracy)),
                          'q95': float(np.percentile(null_accuracy, 95)),
                          'p_value': p_accuracy},
        'attribution': dict(zip(names, attribution.tolist(), strict=True)),
        'attribution_permuted_rank_correlation': {
            'mean': float(null_agreement.mean()),
            'ci95': [float(np.percentile(null_agreement, 2.5)),
                     float(np.percentile(null_agreement, 97.5))]},
        'view_weights': ({f'{v["atlas"]}/{v["statistic"]}': float(w)
                          for v, w in zip(views, weights, strict=True)}
                         if weights is not None else None),
        'n_permutations': args.n_permutations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'hierarchical.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {OUT / "hierarchical.json"}')


if __name__ == '__main__':
    main()
