"""Run the region -> learned-network -> label model on Julia2018, with its nulls.

End to end: region connectivity in, AVGP/NVGP out, the region-to-network
assignment learned jointly with the head (`acnets.deep.region_to_network`).

Three things are reported and all three are priced against permuted labels,
because after manuscript section 3.4.3 nothing in this project reports an
attribution without one:

  accuracy        does the model decode
  attribution     do the learned network edges it leans on depend on the labels
  drift           does the learned parcellation move away from the atlas

Drift is the one specific to this model. The assignment starts at the atlas's
own region-to-network map and is free to leave it, so drift near zero means the
model kept the parcellation it was given and the "learned" part changed nothing;
drift that is high on real labels and equally high on permuted labels means the
assignment is fitting noise. Both readings are visible only by running the null.

`--random-init` is the ablation that matters: if the model only works when
started from the atlas, the atlas is doing the work.

    uv run python scripts/run_learned_parcellation.py            # ~10 min
    uv run python scripts/run_learned_parcellation.py --random-init
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
from acnets.multiverse import accuracy_interval, fixed_splits

warnings.filterwarnings('ignore')

TIMESERIES = Path('data/Julia2018/derivatives/resting_timeseries')
OUT = Path('outputs/learned_parcellation')


def load(atlas: str, statistic: str):
    dataset = xr.open_dataset(TIMESERIES / f'timeseries_{atlas}.nc5',
                              engine='h5netcdf').load()
    timeseries = dataset['timeseries'].transpose(
        'subject', 'timepoint', 'region').values
    subjects = [str(s) for s in dataset['subject'].values]
    labels = np.array([0 if 'AVGP' in s else 1 for s in subjects])

    networks = [str(x) for x in dataset['network'].values]
    names = sorted(set(networks))
    assignment = np.zeros((len(networks), len(names)), dtype='float32')
    for row, network in enumerate(networks):
        assignment[row, names.index(network)] = 1.0
    return timeseries, labels, assignment, names


def fold_connectivity(timeseries, statistic, train_idx, test_idx):
    """Connectivity for both halves from one fit on the training fold."""
    measure = ExtraConnectivityMeasure(kind=statistic, vectorize=False)
    measure.fit(timeseries[train_idx])
    train = np.asarray(measure.transform(timeseries[train_idx]), dtype='float32')
    test = np.asarray(measure.transform(timeseries[test_idx]), dtype='float32')
    return train, test


def fit(model, x, y, epochs, lr, weight_decay):
    """Train end to end, with the assignment exempt from weight decay.

    Decaying the assignment logits pulls the softmax towards uniform, which is
    not a smaller model but a destroyed one: every region ends up a seventh of
    every network, the pooled network matrix loses its structure, and the head
    receives near-identical values. That failure is silent -- it shows up as a
    training accuracy near chance and a mid layer of order 1e-24, not as an
    error -- so the exemption is deliberate rather than incidental.
    """
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
    return model


def evaluate(folds, labels, assignment, args):
    from acnets.deep.region_to_network import RegionToNetworkMLP

    correct = total = 0
    train_accuracies, attributions, drifts = [], [], []
    init = None if args.random_init else torch.tensor(assignment)

    for fold in folds:
        train_idx, test_idx = fold['train_idx'], fold['test_idx']
        x_train = torch.tensor(fold['train'])
        x_test = torch.tensor(fold['test'])
        y_train = torch.tensor(labels[train_idx], dtype=torch.float32)

        torch.manual_seed(args.seed)
        model = RegionToNetworkMLP(
            n_regions=x_train.shape[1], n_networks=assignment.shape[1],
            hidden=tuple(args.hidden), dropout=args.dropout,
            init_assignment=init)
        fit(model, x_train, y_train, args.epochs, args.lr, args.weight_decay)

        with torch.no_grad():
            predicted = (model(x_test).numpy() > 0).astype(int)
            train_predicted = (model(x_train).numpy() > 0).astype(int)
        correct += int((predicted == labels[test_idx]).sum())
        total += len(test_idx)
        train_accuracies.append(float((train_predicted == labels[train_idx]).mean()))
        attributions.append(model.edge_attribution(x_train).numpy())
        drifts.append(model.assignment_drift())

    return {'accuracy': correct / total,
            'train_accuracy': float(np.mean(train_accuracies)),
            'attribution': np.mean(attributions, axis=0),
            'drift': float(np.nanmean(drifts))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', default='dosenbach2010')
    parser.add_argument('--statistic', default='partial correlation')
    parser.add_argument('-B', '--n-permutations', type=int, default=100)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=1)
    parser.add_argument('--hidden', type=int, nargs='+', default=[16, 8])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    timeseries, labels, assignment, names = load(args.atlas, args.statistic)
    n = len(labels)
    splits = fixed_splits(n, args.n_splits, args.n_repeats,
                          random_state=args.seed)
    print(f'{args.atlas}: {timeseries.shape[2]} regions, {n} participants, '
          f'{assignment.shape[1]} networks ({", ".join(names)})')
    print(f'{args.statistic}, {len(splits)} folds, '
          f'{"random" if args.random_init else "atlas"} init, '
          f'B = {args.n_permutations}')

    folds = []
    for train_idx, test_idx in splits:
        train, test = fold_connectivity(timeseries, args.statistic,
                                        train_idx, test_idx)
        folds.append({'train_idx': train_idx, 'test_idx': test_idx,
                      'train': train, 'test': test})

    started = time.time()
    observed = evaluate(folds, labels, assignment, args)
    per_run = time.time() - started
    print(f'\nobserved   test {observed["accuracy"]:.4f}   '
          f'train {observed["train_accuracy"]:.4f}   '
          f'assignment drift {observed["drift"]:.3f}   ({per_run:.0f}s)')
    interval = accuracy_interval(observed['accuracy'] * n, n)
    print(f'           Wilson 95%: [{interval.low:.2f}, {interval.high:.2f}]')
    print(f'estimated total: {per_run * (args.n_permutations + 1) / 60:.1f} min')

    rows, cols = np.triu_indices(assignment.shape[1])
    edge_labels = [f'{names[i]}-{names[j]}' for i, j in zip(rows, cols, strict=True)]
    order = np.argsort(-observed['attribution'])
    print('\ntop learned-network edges:')
    for i in order[:6]:
        print(f'  {edge_labels[i]:22s} {observed["attribution"][i]:.5f}')

    rng = np.random.default_rng(args.seed)
    null_accuracy = np.empty(args.n_permutations)
    null_drift = np.empty(args.n_permutations)
    null_agreement = np.empty(args.n_permutations)
    started = time.time()
    for i in range(args.n_permutations):
        permuted = rng.permutation(labels)
        result = evaluate(folds, permuted, assignment, args)
        null_accuracy[i] = result['accuracy']
        null_drift[i] = result['drift']
        null_agreement[i] = float(np.corrcoef(
            np.argsort(np.argsort(-result['attribution'])),
            np.argsort(np.argsort(-observed['attribution'])))[0, 1])
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - started) / (i + 1)
            print(f'  {i + 1}/{args.n_permutations} '
                  f'({rate:.1f}s each, '
                  f'~{rate * (args.n_permutations - i - 1) / 60:.1f} min left)')

    p_accuracy = float((1.0 + np.sum(null_accuracy >= observed['accuracy']))
                       / (args.n_permutations + 1.0))
    print(f'\naccuracy     observed {observed["accuracy"]:.4f}   '
          f'null median {np.median(null_accuracy):.4f}   '
          f'95th {np.percentile(null_accuracy, 95):.4f}   p = {p_accuracy:.4f}')
    print(f'drift        observed {observed["drift"]:.3f}   '
          f'null median {np.median(null_drift):.3f}   '
          f'(equal drift on real and shuffled labels means the assignment is '
          f'fitting noise)')
    print(f'attribution  rank correlation with permuted-label runs: '
          f'{null_agreement.mean():+.3f} '
          f'[{np.percentile(null_agreement, 2.5):+.3f}, '
          f'{np.percentile(null_agreement, 97.5):+.3f}]')
    print('             near zero = label-dependent; near one = not')

    report = {
        'atlas': args.atlas, 'statistic': args.statistic, 'n': int(n),
        'n_folds': len(splits), 'random_init': args.random_init,
        'epochs': args.epochs, 'hidden': args.hidden,
        'observed_accuracy': observed['accuracy'],
        'train_accuracy': observed['train_accuracy'],
        'wilson95': [interval.low, interval.high],
        'assignment_drift': observed['drift'],
        'accuracy_null': {'median': float(np.median(null_accuracy)),
                          'q95': float(np.percentile(null_accuracy, 95)),
                          'p_value': p_accuracy},
        'drift_null_median': float(np.median(null_drift)),
        'attribution': dict(zip(edge_labels, observed['attribution'].tolist(),
                                strict=True)),
        'attribution_permuted_rank_correlation': {
            'mean': float(null_agreement.mean()),
            'ci95': [float(np.percentile(null_agreement, 2.5)),
                     float(np.percentile(null_agreement, 97.5))]},
        'n_permutations': args.n_permutations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '_random_init' if args.random_init else ''
    (OUT / f'learned_parcellation{suffix}.json').write_text(
        json.dumps(report, indent=2))
    print(f'\nwrote {OUT / f"learned_parcellation{suffix}.json"}')


if __name__ == '__main__':
    main()
