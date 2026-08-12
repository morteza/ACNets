"""Do the three networks replicate across datasets — and across modality?

Section 15 reaches the project's highest accuracy by restricting functional
connectivity to FPN, CON and SMN. It was once recorded as the only
corrected-significant result; §15.1 withdraws that, because those three networks
were read off this dataset's own attribution ranking (§3.1) and no family
declared afterwards can price them. That makes the question below more
important, not less: a restriction chosen from one sample has exactly one honest
test, which is another sample.

**The direct test is impossible, and that limitation is structural.** gong2015
is T1 only -- there is no resting-state fMRI in it -- so "does the FPN/CON/SMN
*connectivity* result replicate in gong2015" cannot be asked at all, in either
direction. No amount of analysis fixes a modality that was never acquired.

What can be asked is the cross-modal version: **if those three networks are
where action-game experience leaves its mark, does restricting *grey-matter
volume* to them help too?** That is a weaker test -- structure and function are
different measurements, and structure has been null everywhere in this project
(sections 20, 21) -- but it is a real one, it runs in both datasets, and it is
the only test here that is not circular: the networks come from Julia2018's
function, so gong2015's structure has never seen them. A positive result would
be evidence that the networks are special rather than fitted.

The mapping from Dosenbach's networks to Schaefer's Yeo-7 labels is fixed in
advance on anatomical grounds, not chosen to maximise anything:

    FPN  fronto-parietal control  ->  Cont
    CON  cingulo-opercular        ->  SalVentAttn
    SMN  sensorimotor             ->  SomMot

Each dataset is tested three ways: restricted to those three networks, on all
200 parcels, and restricted to the three *complementary* networks (Default, Vis,
Limbic) as a control. If the discovered networks are special, restricted should
beat both. If restricted merely beats all-parcels, that is dimensionality, not
localisation -- which is why the complement is there.

    uv run python scripts/run_network_replication.py --n-permutations 1000
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

from acnets.multiverse import accuracy_interval  # noqa: E402

OUT = Path('outputs/network_replication')
ATLAS_DIR = Path('data/atlases/prepared')

# fixed before running, on anatomical correspondence
# the Yeo equivalents of the triple discovered in this dataset (3.1)
DISCOVERED_YEO = ('Cont', 'SalVentAttn', 'SomMot')
CONTROL_YEO = ('Default', 'Vis', 'Limbic')


def dataset(name: str, path: Path, label_of) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_csv(path)
    columns = [c for c in table.columns
               if c not in ('subject', 'total_gm_mm3', 'brain_mask_cm3')]
    y = np.array([label_of(str(s)) for s in table['subject']])
    return table[columns].to_numpy(float), y, columns


def loocv(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[1] < 2:
        return float('nan')
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
    parser.add_argument('--atlas', default='schaefer200')
    parser.add_argument('--n-permutations', type=int, default=1000)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ATLAS_DIR / f'{args.atlas}_labels.csv')
    network_of = {str(row.label): row.network for row in labels.itertuples()}

    datasets = {
        'gong2015': dataset(
            'gong2015',
            Path(f'data/gong2015/derivatives/gmv_gong2015_{args.atlas}.csv'),
            lambda s: 0 if s.startswith('expert') else 1),
        'julia2018': dataset(
            'julia2018',
            Path(f'data/Julia2018/derivatives/gmv_julia2018_{args.atlas}.csv'),
            lambda s: 0 if 'AVGP' in s else 1),
    }

    payload = {}
    for name, (x, y, columns) in datasets.items():
        discovered = [i for i, c in enumerate(columns)
                  if network_of.get(c) in DISCOVERED_YEO]
        control = [i for i, c in enumerate(columns)
                   if network_of.get(c) in CONTROL_YEO]
        print(f'\n{name}: n={len(y)}  '
              f'({int((y == 0).sum())} high-exposure, {int((y == 1).sum())} low)')
        print(f'  discovered networks {"+".join(DISCOVERED_YEO)}: {len(discovered)} parcels')
        print(f'  control networks {"+".join(CONTROL_YEO)}: {len(control)} parcels')

        blocks = {'discovered-networks': x[:, discovered],
                  'all-parcels': x,
                  'control-networks': x[:, control]}
        observed = {k: loocv(v, y) for k, v in blocks.items()}

        rng = np.random.default_rng(args.seed)
        null = np.array(Parallel(n_jobs=args.n_jobs)(
            delayed(loocv)(blocks['discovered-networks'], rng.permutation(y))
            for _ in range(args.n_permutations)))
        p = float((1 + (null >= observed['discovered-networks']).sum())
                  / (len(null) + 1))

        for key, value in observed.items():
            interval = accuracy_interval(value * len(y), len(y))
            # NOT a pre-registered restriction -- it was discovered in
            # Julia2018 (manuscript 3.1). Marked so the row is readable, not to
            # imply it was declared in advance.
            marker = ('  <- the restriction discovered in Julia2018'
                      if 'discovered' in key else '')
            print(f'    {key:20s} {value:.3f}  '
                  f'[{interval.low:.2f}, {interval.high:.2f}]{marker}')
        print(f'    discovered-networks vs its own null: median '
              f'{np.median(null):.3f}, p = {p:.4f}'
              f'{"   REPLICATES" if p < 0.05 else "   does not replicate"}')

        payload[name] = {'n': len(y), 'observed': observed,
                         'p_discovered': p, 'null_median': float(np.median(null)),
                         'n_discovered_parcels': len(discovered),
                         'n_control_parcels': len(control)}

    (OUT / f'replication_{args.atlas}.json').write_text(json.dumps({
        'atlas': args.atlas,
        'discovered_networks': list(DISCOVERED_YEO),
        'control_networks': list(CONTROL_YEO),
        'modality': 'grey-matter volume',
        'limitation': 'gong2015 has no resting-state fMRI, so the functional '
                      'result of section 15 cannot be tested in it directly. '
                      'This is the cross-modal analogue.',
        'datasets': payload,
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
