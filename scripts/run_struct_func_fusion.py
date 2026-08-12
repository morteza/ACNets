"""Combine corrected structural volumes with functional connectivity.

Section 6 already tested this and found structural features *degraded* every
combination they entered (connectivity 0.666 alone, 0.577 with structure). But
that used fmriprep's `aparc+aseg` labels, computed by a different tool from a
different segmentation, and the gong2015 arm's volumes were later found to be
skull-contaminated and read through an unusable sphere atlas (§18, §19).

The structural features are now rebuilt: one ANTs pipeline, verified brain
extraction, volumetric atlases whose cross-dataset profile agreement is +0.87 to
+0.94. That is a good enough change to justify asking again rather than citing
the old answer.

Both modalities are measured on the same 32 Julia2018 participants, so nothing
is lost to a join. The functional side uses the section 15 configuration --
precision connectivity restricted to FPN, CON and SMN -- because that is the
best-supported functional model in the project, not because it was the best of a
sweep run here.

The specification set is fixed before running and deliberately small: functional
alone, structural alone for three atlases, and the two concatenated for three
atlases. Seven cells. A family-wise null covers all of them, because the whole
point of the exercise is to ask whether *any* of them beats functional alone.

    uv run python scripts/run_struct_func_fusion.py --n-permutations 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.covariance import OAS  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse import accuracy_interval, max_stat_permutation  # noqa: E402
from acnets.multiverse.runner import edge_names  # noqa: E402
from run_agreement import aggregate, group_labels, load_atlas  # noqa: E402
from run_reliability_transfer import connectivity  # noqa: E402

OUT = Path('outputs/fusion')
JULIA = Path('data/Julia2018/derivatives')
ATLASES = ('aal', 'harvard-oxford', 'schaefer200')
# discovered in this dataset (manuscript 3.1), not declared in advance
DISCOVERED_NETWORKS = ('FPN', 'CON', 'SMN')


def functional(seed: int):
    """Section 15's configuration: precision, restricted to three networks."""
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, seed)
    names = edge_names(nodes['network'], False)
    x = connectivity(timeseries['network'], 'precision')
    pairs = [tuple(n.split(' ↔ ')) for n in names]
    keep = [i for i, (a, b) in enumerate(pairs)
            if a in DISCOVERED_NETWORKS and b in DISCOVERED_NETWORKS]
    subjects = [str(s) for s in dataset['subject'].values]
    return pd.DataFrame(x[:, keep], index=subjects), y, subjects


def structural(atlas: str, subjects: list[str]) -> pd.DataFrame | None:
    path = JULIA / f'gmv_julia2018_{atlas}.csv'
    if not path.exists():
        return None
    table = pd.read_csv(path).set_index('subject')
    table = table.drop(columns=[c for c in ('total_gm_mm3', 'brain_mask_cm3')
                                if c in table.columns])
    table.index = [str(i) for i in table.index]
    missing = [s for s in subjects if s not in table.index]
    if missing:
        print(f'  {atlas}: {len(missing)} participants lack structural data')
        return None
    return table.loc[subjects]


def loocv(x: np.ndarray, y: np.ndarray) -> float:
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        model = LinearDiscriminantAnalysis(solver='lsqr',
                                           covariance_estimator=OAS())
        model.fit(scaler.transform(x[train]), y[train])
        predictions[test] = model.predict(scaler.transform(x[test]))
    return float((predictions == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    func, y, subjects = functional(args.seed)
    print(f'{len(y)} participants; functional {func.shape[1]} edges '
          f'(precision, {"+".join(DISCOVERED_NETWORKS)})')

    blocks: dict[str, np.ndarray] = {'functional': func.to_numpy(float)}
    for atlas in ATLASES:
        struct = structural(atlas, subjects)
        if struct is None:
            continue
        print(f'  structural/{atlas}: {struct.shape[1]} regions')
        blocks[f'structural/{atlas}'] = struct.to_numpy(float)
        # concatenate after each block is z-scored in-fold by the pipeline, so
        # the 6 functional features are not drowned by 166 structural ones
        blocks[f'fusion/{atlas}'] = np.hstack(
            [func.to_numpy(float), struct.to_numpy(float)])

    def evaluate(labels: np.ndarray) -> dict[str, float]:
        return {name: loocv(x, labels) for name, x in blocks.items()}

    observed = evaluate(y)
    print()
    for name, value in sorted(observed.items(), key=lambda kv: -kv[1]):
        interval = accuracy_interval(value * len(y), len(y))
        print(f'  {name:26s} {value:.3f}   '
              f'[{interval.low:.2f}, {interval.high:.2f}]')

    rng = np.random.default_rng(args.seed)
    null = np.array(Parallel(n_jobs=args.n_jobs)(
        delayed(lambda: [evaluate(rng.permutation(y))[k] for k in observed])()
        for _ in range(args.n_permutations)))
    result = max_stat_permutation(observed, null)

    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}:')
    print(f'  observed max        {result.observed_max:.3f}')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  null max, 95th pct  {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')

    gain = max(v for k, v in observed.items() if k.startswith('fusion')) \
        - observed['functional']
    print(f'\n  best fusion vs functional alone: {gain:+.3f}')

    (OUT / 'fusion.json').write_text(json.dumps({
        'n': len(y), 'observed': observed,
        'fusion_gain_over_functional': float(gain),
        **result.summary(),
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
