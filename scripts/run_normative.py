"""Normative modelling: is an AVGP's connectome *atypical* rather than different?

Every other analysis in this project asks whether the two groups differ in the
value of some connectivity feature. This one asks a different question: relative
to 220 external participants who were never labelled, how unusual is each
participant's connectome -- and does *that* separate the groups?

The distinction is not cosmetic. A deviation score is not another view of the
same covariance matrix, which is why the other six statistics could not find
what each other missed. Two participants with identical CON-SMN coupling can
have different deviation scores, because the score depends on the whole vector
and on how an external population covaries. It is the one feature family
identified in `docs/paper/decodability_ceiling.md` as genuinely untested.

Design:

  1. Train a denoising diffusion model on MPI-LEMON connectivity vectors --
     220 participants cut into 124-timepoint windows, matching the target run
     length, giving ~1,100 training vectors (`acnets.deep.VectorDiffusion`).
  2. Score each Julia2018 participant by per-edge denoising error, averaged over
     many noise draws. The score lives in the same 21-edge vocabulary as every
     attribution in the paper.
  3. Classify the groups from the deviation vectors, on the same
     label-independent folds used everywhere else.
  4. Compare against three controls that matter:
       raw          the connectivity values themselves -- the paper's baseline
       gaussian     Mahalanobis deviation from the same LEMON population, i.e.
                    the classical normative model. If the diffusion model does
                    not beat this, its capacity bought nothing.
       shuffled     a diffusion model trained on LEMON vectors with each edge
                    independently permuted across participants, destroying the
                    covariance structure while preserving every marginal. If the
                    real model does not beat this, the score is reading marginals
                    rather than population structure.
  5. Run the family-wise permutation null over the whole comparison, because
     this is five more specifications on the same 32 people and reporting the
     best of them uncorrected is exactly the error the rest of the paper avoids.

Leakage: LEMON and Julia2018 share no participants, LEMON carries no labels, and
the scoring function has no parameter through which one could be supplied.

    uv run python scripts/run_normative.py
    uv run python scripts/run_normative.py --n-permutations 200 --epochs 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Before `import torch`, not after: XGBoost must bind its OpenMP runtime first
# or a later XGBoost fit segfaults the process with no traceback. Importing
# acnets.deep does this too, but only once torch is already loaded here, which
# is too late. See acnets._openmp.
from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure  # noqa: E402
from acnets.deep.diffusion import GaussianNormative, VectorDiffusion  # noqa: E402
from acnets.deep.training import MetricLogger, Trainer  # noqa: E402
from acnets.multiverse import (  # noqa: E402
    accuracy_interval,
    fixed_splits,
    max_stat_permutation,
)
from acnets.multiverse.runner import (  # noqa: E402
    _make_classifier,
    edge_names,
    vectorize,
)
from acnets.multiverse.transfer import windowed  # noqa: E402
from acnets.pipeline import TimeseriesAggregator  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402

LEMON = Path('data/LEMON')
OUT = Path('outputs/normative')
STATISTIC = 'partial correlation'
CLASSIFIERS = ('svm', 'xgboost', 'mlp')


def connectivity(timeseries: np.ndarray) -> np.ndarray:
    """Vectorised connectivity, computed per participant with no group fitting."""
    measure = ExtraConnectivityMeasure(kind=STATISTIC, vectorize=False)
    measure.fit(timeseries)
    return vectorize(measure.transform(timeseries), False)


def lemon_vectors(nodes: list[str], window_length: int) -> np.ndarray:
    """LEMON connectivity vectors, windowed to the target's run length."""
    path = LEMON / 'timeseries_dosenbach2010.nc'
    if not path.exists():
        raise SystemExit(f'{path} missing; normative modelling needs LEMON')

    dataset = xr.open_dataset(path, engine='h5netcdf').load()
    aggregated = TimeseriesAggregator(strategy='network').fit_transform(dataset)
    values = aggregated['timeseries'].transpose(
        'subject', 'timepoint', 'network').values
    lemon_nodes = [str(n) for n in aggregated['network'].values]
    if lemon_nodes != nodes:
        raise SystemExit(f'LEMON vocabulary {lemon_nodes} != target {nodes}')

    windows = windowed(values, window_length)
    print(f'LEMON: {len(windows)} windows of {window_length} timepoints '
          f'from {dataset.sizes["subject"]} participants')
    return connectivity(windows)


def train_diffusion(vectors: np.ndarray, epochs: int, seed: int,
                    quiet: bool = True) -> tuple[VectorDiffusion, StandardScaler]:
    """Fit the density on the external population. Never sees a label."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(vectors)
    scaled = scaler.transform(vectors).astype(np.float32)

    model = VectorDiffusion(n_features=scaled.shape[1], hidden=128,
                            n_layers=3, n_steps=200)
    loader = DataLoader(TensorDataset(torch.tensor(scaled)),
                        batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    trainer = Trainer(max_epochs=epochs, lr=1e-3,
                      logger=MetricLogger('normative', enabled=False))
    trainer.fit(model, loader)
    if not quiet:
        print(f'  diffusion loss {trainer.history[0]["loss/train"]:.4f} -> '
              f'{trainer.history[-1]["loss/train"]:.4f}')
    return model, scaler


def evaluate(x: np.ndarray, y: np.ndarray, splits, classifier: str) -> float:
    folds = []
    for train, test in splits:
        scaler = StandardScaler().fit(x[train])
        model = _make_classifier(classifier, 0)
        model.fit(scaler.transform(x[train]), y[train])
        folds.append(float((model.predict(scaler.transform(x[test])) == y[test]).mean()))
    return float(np.mean(folds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n-draws', type=int, default=256)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--n-splits', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, args.seed)
    network_nodes = nodes['network']
    names = edge_names(network_nodes, False)

    target = connectivity(timeseries['network'])
    print(f'target: {target.shape[0]} participants x {target.shape[1]} edges')

    external = lemon_vectors(network_nodes, dataset.sizes['timepoint'])

    # ---- the three normative models ---------------------------------------
    started = time.perf_counter()
    print('\ntraining on LEMON (no labels anywhere in this step):')
    model, scaler = train_diffusion(external, args.epochs, args.seed, quiet=False)
    diffusion_scores = model.deviation(
        scaler.transform(target).astype(np.float32),
        n_draws=args.n_draws, seed=args.seed)

    gaussian = GaussianNormative().fit(external)
    gaussian_scores = gaussian.deviation(target)

    # marginals preserved, joint structure destroyed
    rng = np.random.default_rng(args.seed)
    shuffled_external = np.column_stack(
        [rng.permutation(external[:, j]) for j in range(external.shape[1])])
    shuffled_model, shuffled_scaler = train_diffusion(
        shuffled_external, args.epochs, args.seed)
    shuffled_scores = shuffled_model.deviation(
        shuffled_scaler.transform(target).astype(np.float32),
        n_draws=args.n_draws, seed=args.seed)
    print(f'  trained and scored in {time.perf_counter() - started:.0f}s')

    feature_sets = {
        'raw': target,
        'diffusion': diffusion_scores,
        'gaussian': gaussian_scores,
        'diffusion-shuffled': shuffled_scores,
        'raw+diffusion': np.hstack([target, diffusion_scores]),
    }

    splits = fixed_splits(len(y), args.n_splits, args.n_repeats, args.seed)
    observed, rows = {}, []
    print(f'\naccuracy over {args.n_splits} x {args.n_repeats} folds:')
    for family, x in feature_sets.items():
        for classifier in CLASSIFIERS:
            key = f'{family}/{classifier}'
            accuracy = evaluate(x, y, splits, classifier)
            interval = accuracy_interval(accuracy * len(y), len(y))
            observed[key] = accuracy
            rows.append({'family': family, 'classifier': classifier,
                         'accuracy': accuracy, 'n_features': int(x.shape[1]),
                         'wilson': [interval.low, interval.high]})
            print(f'  {family:20s} {classifier:8s} {accuracy:.3f}  '
                  f'[{interval.low:.2f}, {interval.high:.2f}]')

    # ---- does the deviation score correlate with the raw values? ----------
    # If it does perfectly, it is not a new quantity and nothing above matters.
    from scipy.stats import spearmanr
    # Six of the 21 columns are the partial-correlation diagonal, fixed at 1 by
    # construction, so a correlation with them is undefined. They are kept in
    # the feature matrix because the rest of the paper uses the same 21-edge
    # vocabulary, but they are skipped here.
    varying = [j for j in range(target.shape[1]) if np.ptp(target[:, j]) > 1e-12]
    correlations = [abs(spearmanr(target[:, j], diffusion_scores[:, j]).statistic)
                    for j in varying]
    print(f'\n|Spearman| between each edge and its deviation score, over the '
          f'{len(varying)} non-constant edges: '
          f'median {np.median(correlations):.2f}, max {np.max(correlations):.2f}')

    # ---- family-wise null over everything just tried ----------------------
    print(f'\nfamily-wise null, B = {args.n_permutations} '
          f'(labels permuted outside everything):')
    null = np.empty((args.n_permutations, len(observed)))
    rng = np.random.default_rng(args.seed)
    for i in range(args.n_permutations):
        permuted = rng.permutation(y)
        scores = {f'{family}/{classifier}': evaluate(x, permuted, splits, classifier)
                  for family, x in feature_sets.items()
                  for classifier in CLASSIFIERS}
        null[i] = [scores[k] for k in observed]
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{args.n_permutations}', flush=True)

    result = max_stat_permutation(observed, null)
    best = max(observed, key=observed.get)
    print(f'\n  best                {observed[best]:.3f}  ({best})')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  critical value (95) {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')
    n_significant = sum(v >= result.critical_value for v in observed.values())
    print(f'  {n_significant} of {len(observed)} specifications clear it')

    (OUT / 'normative.json').write_text(json.dumps({
        'statistic': STATISTIC,
        'n_subjects': len(y),
        'n_lemon_windows': len(external),
        'epochs': args.epochs, 'n_draws': args.n_draws,
        'edge_names': names,
        'results': rows,
        'deviation_vs_raw_spearman': {
            'n_edges_compared': len(varying),
            'median': float(np.median(correlations)),
            'max': float(np.max(correlations))},
        **result.summary(),
        'n_significant': int(n_significant),
    }, indent=2, default=float))
    np.savez(OUT / 'scores.npz', diffusion=diffusion_scores,
             gaussian=gaussian_scores, shuffled=shuffled_scores, raw=target)
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
