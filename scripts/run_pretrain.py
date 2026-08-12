"""Pretrain the timeseries encoder on LEMON. No labels exist here.

This is the pretraining half of the pre-specified primary (plan.md C2/C7): a
sequence autoencoder over network-scale resting timeseries, trained to
reconstruct 124-timepoint windows of MPI-LEMON. The cached LEMON file carries
no participant metadata at all, so there is nothing to leak even by accident;
the model learns what generic resting coactivation looks like, nothing else.

The reconstruction metrics this prints are the only quantities consulted when
freezing the primary's architecture in `docs/paper/prereg_primary.md`. That is
the boundary the prereg depends on: architecture choices may look at LEMON
reconstruction, never at Julia2018 accuracy.

Usage:
    uv run python scripts/run_pretrain.py            # writes checkpoints/
    uv run python scripts/run_pretrain.py --epochs 5 # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from acnets.deep.seq2seq import Seq2SeqAE
from acnets.deep.timepoint_ae import TimepointAE
from acnets.deep.training import MetricLogger, Trainer, checkpoint_id
from acnets.multiverse import windowed
from acnets.pipeline import TimeseriesAggregator

LEMON = Path('data/LEMON/timeseries_dosenbach2010.nc')
OUT = Path('outputs/pretrain')


class ReconstructionWrapper(torch.nn.Module):
    """Adapts Seq2SeqAE's (h, x_recon, loss) contract to Trainer's compute_loss."""

    def __init__(self, autoencoder: Seq2SeqAE):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        return self.autoencoder(x)

    def compute_loss(self, batch, device):
        x = batch[0].to(device)
        _, _reconstruction, loss = self.autoencoder(x)
        # variance-normalised error: 1.0 means "no better than predicting the
        # mean", which windows are z-scored to, so this is directly readable
        relative = float(loss.detach()) / float(x.var())
        return loss, {'relative_error': relative}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=('timepoint', 'seq2seq'),
                        default='timepoint',
                        help='seq2seq is kept as the rejected candidate: its '
                             'held-out reconstruction never beat the mean '
                             'predictor (see outputs/pretrain/)')
    parser.add_argument('--latent', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window', type=int, default=124,
                        help="Julia2018's run length; fixed by the target")
    parser.add_argument('--stride', type=int, default=62,
                        help='overlapping windows double the sample')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = xr.open_dataset(LEMON, engine='h5netcdf').load()
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    values = aggregated['timeseries'].transpose('subject', 'timepoint', 'network').values
    networks = [str(n) for n in aggregated['network'].values]

    windows = windowed(values, args.window, args.stride)
    print(f'LEMON: {values.shape[0]} subjects x {values.shape[1]} timepoints '
          f'-> {windows.shape[0]} windows of {args.window} x {len(networks)} '
          f'({networks})')

    # held-out reconstruction: split by SUBJECT, not by window, so overlapping
    # windows of one person never straddle the boundary
    n_subjects = values.shape[0]
    windows_per_subject = windows.shape[0] // n_subjects
    rng = np.random.default_rng(args.seed)
    subject_order = rng.permutation(n_subjects)
    n_val = n_subjects // 10
    val_subjects = set(subject_order[:n_val].tolist())
    subject_of_window = np.tile(np.arange(n_subjects), windows_per_subject)
    is_val = np.isin(subject_of_window, list(val_subjects))

    to_loader = lambda mask, shuffle: DataLoader(  # noqa: E731
        TensorDataset(torch.tensor(windows[mask], dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=shuffle)

    hparams = {'arch': args.arch, 'input_size': len(networks),
               'latent': args.latent, 'window': args.window,
               'stride': args.stride, 'epochs': args.epochs,
               'batch_size': args.batch_size, 'seed': args.seed,
               'source': 'LEMON/dosenbach2010/network'}
    run_id = checkpoint_id(hparams)
    run_name = f'lemon_{args.arch}_{run_id}'

    if args.arch == 'timepoint':
        # the AE sees flat timepoints; batches of windows keep the loader shape
        model = TimepointAE(len(networks), latent=args.latent)
    else:
        model = ReconstructionWrapper(Seq2SeqAE(len(networks), args.latent))
    trainer = Trainer(max_epochs=args.epochs,
                      checkpoint_dir=Path('checkpoints') / run_name,
                      logger=MetricLogger(run_name,
                                          root='outputs/pretrain/runs'))
    trainer.fit(model, to_loader(~is_val, True), to_loader(is_val, False))

    first, last = trainer.history[0], trainer.history[-1]
    print('\nreconstruction, relative to predicting the window mean:')
    print(f'  epoch 0   train {first["relative_error/train"]:.3f}   '
          f'val {first.get("relative_error/val", float("nan")):.3f}')
    print(f'  epoch {last["epoch"]:<3d} train {last["relative_error/train"]:.3f}   '
          f'val {last.get("relative_error/val", float("nan")):.3f}')
    print(f'  ({n_val} held-out subjects, split by subject)')

    payload = {'hparams': hparams, 'checkpoint_id': run_id,
               'checkpoint': f'checkpoints/{run_name}/last.pt',
               'n_windows': int(windows.shape[0]),
               'val_subjects': sorted(val_subjects),
               'history': trainer.history}
    (OUT / f'{run_name}.json').write_text(
        json.dumps(payload, indent=2, default=float))
    print(f'\ncheckpoint: checkpoints/{run_name}/last.pt')
    print(f'metrics:    {OUT / f"{run_name}.json"}')


if __name__ == '__main__':
    main()
