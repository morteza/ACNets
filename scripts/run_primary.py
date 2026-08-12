"""Execute the pre-specified confirmatory primary. Exactly once, as declared.

Everything this script does is frozen in `docs/paper/prereg_primary.md` and
`acnets.multiverse.registry.PRIMARY_CONFIRMATORY`; it refuses to run without
the declaration in place, and it takes no arguments that could change the
analysis — the only flags control the permutation count (for staging the null
in chunks) and worker processes.

    uv run python scripts/run_primary.py                    # observed + B=1000
    uv run python scripts/run_primary.py --n-permutations 0 # observed only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from joblib import Parallel, delayed

from acnets.deep.timepoint_ae import LatentConnectivityClassifier, TimepointAE
from acnets.deep.training import load_checkpoint
from acnets.explain.haufe import haufe_transform
from acnets.multiverse import accuracy_interval, fixed_splits, permutation_pvalue
from acnets.multiverse.registry import primary_confirmatory
from acnets.pipeline import TimeseriesAggregator

DECLARATION = Path('docs/paper/prereg_primary.md')
DATA = Path('data/Julia2018/timeseries_dosenbach2010.nc5')
OUT = Path('outputs/primary')

PRIMARY = primary_confirmatory()


def load_target() -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = xr.open_dataset(DATA, engine='h5netcdf').load()
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    values = aggregated['timeseries'].transpose('subject', 'timepoint', 'network').values
    networks = [str(n) for n in aggregated['network'].values]

    # per-channel z-scoring over time, the same operation the LEMON windows
    # received in pretraining
    mean = values.mean(axis=1, keepdims=True)
    sd = values.std(axis=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    values = (values - mean) / sd

    labels = np.array([0 if str(s).startswith('AVGP') else 1
                       for s in dataset['subject'].values])
    return values.astype(np.float32), labels, networks


def build_model() -> LatentConnectivityClassifier:
    encoder_spec = PRIMARY['encoder']
    autoencoder = TimepointAE(n_channels=encoder_spec['n_channels'],
                              latent=encoder_spec['latent'],
                              hidden=encoder_spec['hidden'])
    load_checkpoint(autoencoder, PRIMARY['checkpoint'])
    return LatentConnectivityClassifier(autoencoder)


def _train_phase(model, x, y, parameters, epochs, lr, batch_size, generator):
    optimizer = torch.optim.Adam(parameters, lr=lr)
    n = x.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()


def finetune_fold(x: np.ndarray, y: np.ndarray,
                  train_idx: np.ndarray, test_idx: np.ndarray,
                  fold: int) -> tuple[float, np.ndarray]:
    """One fold, from the frozen checkpoint, per the declaration."""
    torch.set_num_threads(1)  # workers must not oversubscribe the cores
    plan = PRIMARY['finetune']
    seed = plan['seed_base'] + fold
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = build_model()
    x_train = torch.tensor(x[train_idx])
    y_train = torch.tensor(y[train_idx])

    model.set_encoder_frozen(True)
    _train_phase(model, x_train, y_train, model.head.parameters(),
                 plan['head_only_epochs'], plan['head_only_lr'],
                 plan['batch_size'], generator)
    model.set_encoder_frozen(False)
    _train_phase(model, x_train, y_train, model.parameters(),
                 plan['joint_epochs'], plan['joint_lr'],
                 plan['batch_size'], generator)

    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(x[test_idx])).argmax(1).numpy()
        accuracy = float((predicted == y[test_idx]).mean())
        # exploratory attribution: Haufe pattern of the head's class contrast
        # over the 14 latent-summary features, computed on the training fold
        features = model.features(x_train).numpy()
        weights = (model.head.weight[1] - model.head.weight[0]).numpy()
    pattern = haufe_transform(features, weights)
    return accuracy, pattern


def mean_accuracy(x, labels, splits, n_jobs) -> tuple[float, np.ndarray]:
    results = Parallel(n_jobs=n_jobs)(
        delayed(finetune_fold)(x, labels, train, test, fold)
        for fold, (train, test) in enumerate(splits))
    accuracies = np.array([a for a, _ in results])
    patterns = np.stack([p for _, p in results])
    return float(accuracies.mean()), patterns


def null_statistic(x, permuted, splits) -> float:
    """One permutation's mean accuracy; folds run serially inside the worker."""
    accuracies = [finetune_fold(x, permuted, train, test, fold)[0]
                  for fold, (train, test) in enumerate(splits)]
    return float(np.mean(accuracies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int,
                        default=PRIMARY['test']['n_permutations'])
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()

    if not DECLARATION.exists():
        raise SystemExit(f'{DECLARATION} is missing; the primary may only be '
                         'run against its frozen declaration')
    OUT.mkdir(parents=True, exist_ok=True)

    x, labels, networks = load_target()
    cv = PRIMARY['cv']
    if len(labels) != cv['n_subjects']:
        raise SystemExit(f'expected {cv["n_subjects"]} subjects, found {len(labels)}')
    splits = fixed_splits(cv['n_subjects'], cv['n_splits'], cv['n_repeats'],
                          cv['random_state'])

    print(f'primary: {PRIMARY["name"]}')
    print(f'  checkpoint {PRIMARY["checkpoint_id"]}, {len(splits)} folds, '
          f'declaration {DECLARATION}')

    started = time.perf_counter()
    observed, patterns = mean_accuracy(x, labels, splits, args.n_jobs)
    wilson = accuracy_interval(observed * len(labels), len(labels))
    print(f'\nobserved mean accuracy: {observed:.3f}')
    print(f'  sampling CI at n = {len(labels)}: {wilson}')
    print(f'  ({time.perf_counter() - started:.0f}s for the observed statistic)')

    feature_names = ([f'latent{i}_mean' for i in range(4)]
                     + [f'latentcov_{i}{j}' for i in range(4) for j in range(i, 4)])
    attribution = dict(zip(feature_names, patterns.mean(axis=0).tolist(),
                           strict=True))

    null = None
    p_value = None
    if args.n_permutations > 0:
        rng = np.random.default_rng(cv['random_state'])
        permutations = [rng.permutation(labels) for _ in range(args.n_permutations)]
        print(f'\nnull: {args.n_permutations} permutations, same splits and '
              f'seeds, folds serial within each worker')
        started = time.perf_counter()
        null = Parallel(n_jobs=args.n_jobs, verbose=1)(
            delayed(null_statistic)(x, permuted, splits)
            for permuted in permutations)
        null = np.asarray(null)
        p_value = permutation_pvalue(observed, null)
        print(f'  null mean {null.mean():.3f}, 95th percentile '
              f'{np.quantile(null, 0.95):.3f}')
        print(f'  p = {p_value:.4f} (floor {1 / (args.n_permutations + 1):.4f}, '
              f'alpha {PRIMARY["test"]["alpha"]})')
        print(f'  ({(time.perf_counter() - started) / 60:.1f} min)')
        np.save(OUT / 'null_mean.npy', null)

    (OUT / 'primary.json').write_text(json.dumps({
        'declaration': str(DECLARATION),
        'primary': PRIMARY,
        'networks': networks,
        'observed_mean_accuracy': observed,
        'wilson_ci': [wilson.low, wilson.high],
        'n_permutations': args.n_permutations,
        'p_value': p_value,
        'attribution_haufe_latent': attribution,
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
