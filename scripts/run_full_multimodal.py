"""Everything at once: all modalities, atlases, connectivities and models.

The kitchen-sink analysis, run properly. Four modalities, four connectivity
statistics, three structural atlases, seven model families including a neural
network, and every modality combination -- then **one family-wise permutation
null over the entire grid**.

That last part is the whole point. Any sufficiently large search produces a large
maximum; the question is whether the maximum beats what the *same* search
produces from permuted labels. Running this without the null would be section
10's rung 1, and worth +0.15 of illusion.

Modalities, and what each licenses:

  functional   4 connectivity statistics x {all 21 edges, FPN/CON/SMN only}
  structural   corrected GMV, 3 volumetric atlases (sections 18-20)
  behaviour    accuracy, RT, inverse efficiency -- **near-circular**: gamers
               outperform non-gamers on attention tasks, which is why they were
               recruited, so predicting group from it restates the design
  motion       a nuisance variable that differs by group (section 3.5)

Rows using behaviour or motion are flagged in the output. A high number that
needs either is a claim about the study design, not about the brain.

Only the 25 participants with behavioural records are used throughout, so every
cell is computed on the same people and the accuracies are comparable. This
costs 7 participants against the functional-only analyses elsewhere, which is
why numbers here do not match section 15 exactly.

    uv run python scripts/run_full_multimodal.py --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

from acnets._openmp import preload as _preload_xgboost

_preload_xgboost()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse.classifiers import make_classifier  # noqa: E402
from acnets.multiverse.runner import edge_names  # noqa: E402
from run_agreement import aggregate, load_atlas  # noqa: E402
from run_brain_behavior import behavioural_performance  # noqa: E402
from run_motion_control import load as load_motion  # noqa: E402
from run_reliability_transfer import connectivity  # noqa: E402

OUT = Path('outputs/full_multimodal')
JULIA = Path('data/Julia2018/derivatives')
STATISTICS = ('correlation', 'partial correlation', 'precision', 'tangent')
STRUCT_ATLASES = ('aal', 'harvard-oxford', 'schaefer200')
MODELS = ('lda', 'ridge', 'logistic', 'svm', 'gnb', 'random-forest', 'mlp')
# discovered in this dataset (manuscript 3.1), not declared in advance
DISCOVERED = ('FPN', 'CON', 'SMN')

KIND = {'functional': 'brain', 'structural': 'brain',
        'behaviour': 'near-circular', 'motion': 'nuisance'}


def build(seed: int):
    """{block name: DataFrame}, all on the participants common to every block."""
    dataset = load_atlas('dosenbach2010')
    timeseries, nodes = aggregate(dataset, seed)
    subjects = [str(s) for s in dataset['subject'].values]
    names = edge_names(nodes['network'], False)
    pairs = [tuple(n.split(' ↔ ')) for n in names]
    discovered = [i for i, (a, b) in enumerate(pairs)
                  if a in DISCOVERED and b in DISCOVERED]

    blocks: dict[str, pd.DataFrame] = {}
    for statistic in STATISTICS:
        x = connectivity(timeseries['network'], statistic)
        short = statistic.replace('partial correlation', 'pcorr')
        blocks[f'functional:{short}-all'] = pd.DataFrame(x, index=subjects)
        blocks[f'functional:{short}-discovered'] = pd.DataFrame(x[:, discovered],
                                                            index=subjects)
    for atlas in STRUCT_ATLASES:
        path = JULIA / f'gmv_julia2018_{atlas}.csv'
        if not path.exists():
            continue
        table = pd.read_csv(path).set_index('subject')
        table = table.drop(columns=[c for c in ('total_gm_mm3', 'brain_mask_cm3')
                                    if c in table.columns])
        table.index = [str(i) for i in table.index]
        blocks[f'structural:{atlas}'] = table

    behaviour = behavioural_performance()
    behaviour.index = [str(i) for i in behaviour.index]
    blocks['behaviour:task'] = behaviour

    motion, _ = load_motion()
    motion.index = [str(i) for i in motion.index]
    blocks['motion:confounds'] = motion

    common = set(subjects)
    for frame in blocks.values():
        common &= set(frame.index)
    common = sorted(common)
    labels = np.array([0 if 'AVGP' in s else 1 for s in common])

    return {k: v.loc[common] for k, v in blocks.items()}, labels, common


def loocv(x: np.ndarray, y: np.ndarray, model: str, seed: int = 0) -> float:
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        try:
            fitted = make_classifier(model, random_state=seed)
            fitted.fit(scaler.transform(x[train]), y[train])
            predictions[test] = fitted.predict(scaler.transform(x[test]))
        except Exception:
            return float('nan')
    return float((predictions == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--max-modalities', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    blocks, y, _common = build(args.seed)
    print(f'{len(y)} participants common to every modality '
          f'({int((y == 0).sum())} AVGP, {int((y == 1).sum())} NVGP)')
    for name, frame in blocks.items():
        print(f'  {name:34s} {frame.shape[1]:4d} features')

    names = list(blocks)
    combos = [c for size in range(1, args.max_modalities + 1)
              for c in combinations(names, size)]
    # only combinations that mix *different* modality types are interesting;
    # concatenating two connectivity statistics is a wider feature set, not a
    # multimodal model, and it multiplies the grid for nothing
    combos = [c for c in combos
              if len({n.split(':')[0] for n in c}) == len(c)]
    matrices = {c: np.hstack([blocks[n].to_numpy(float) for n in c])
                for c in combos}
    cells = [(c, m) for c in combos for m in MODELS]
    print(f'\n{len(combos)} modality combinations x {len(MODELS)} models '
          f'= {len(cells)} specifications')

    def evaluate(labels):
        return [loocv(matrices[c], labels, m, args.seed) for c, m in cells]

    started = time.perf_counter()
    observed = np.array(evaluate(y))
    print(f'observed grid in {time.perf_counter() - started:.0f}s')

    rows = []
    for (combo, model), value in zip(cells, observed, strict=True):
        if not np.isfinite(value):
            continue
        kinds = {KIND[n.split(':')[0]] for n in combo}
        rows.append({'modalities': '+'.join(combo), 'model': model,
                     'accuracy': value, 'n_features': matrices[combo].shape[1],
                     'brain_only': kinds <= {'brain'},
                     'uses_behaviour': 'near-circular' in kinds,
                     'uses_motion': 'nuisance' in kinds})
    table = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    table.to_csv(OUT / 'full_multimodal.csv', index=False)

    print('\ntop 10 of the whole grid:')
    for _, r in table.head(10).iterrows():
        flag = ('BRAIN' if r['brain_only']
                else 'behaviour' if r['uses_behaviour'] else 'motion')
        print(f'  {r["accuracy"]:.3f}  {r["modalities"][:52]:52s} '
              f'{r["model"]:14s} {flag}')

    brain = table[table['brain_only']]
    print(f'\nbest overall           {table.iloc[0]["accuracy"]:.3f}')
    if len(brain):
        print(f'best brain-only        {brain.iloc[0]["accuracy"]:.3f}  '
              f'({brain.iloc[0]["modalities"][:50]})')

    started = time.perf_counter()
    print(f'\nfamily-wise null over all {len(cells)} specifications, '
          f'B = {args.n_permutations}:')

    def one(i):
        rng = np.random.default_rng(args.seed + 1 + i)
        return evaluate(rng.permutation(y))

    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(one)(i) for i in range(args.n_permutations)))
    null = np.nan_to_num(null, nan=0.5)
    print(f'  {time.perf_counter() - started:.0f}s')

    null_max = null.max(axis=1)
    best = float(np.nanmax(observed))
    critical = float(np.percentile(null_max, 95))
    p = float((1 + (null_max >= best).sum()) / (len(null_max) + 1))
    print(f'\n  observed max        {best:.3f}')
    print(f'  null max, median    {np.median(null_max):.3f}')
    print(f'  null max, 95th pct  {critical:.3f}')
    print(f'  p                   {p:.4f}'
          f'{"   SIGNIFICANT" if p < 0.05 else ""}')
    print(f'  cells clearing the critical value: '
          f'{int((observed[np.isfinite(observed)] >= critical).sum())}')

    (OUT / 'full_multimodal.json').write_text(json.dumps({
        'n': len(y), 'n_specifications': len(cells),
        'observed_max': best, 'critical_value_95': critical, 'p_value': p,
        'null_max_median': float(np.median(null_max)),
        'best_overall': table.iloc[0].to_dict(),
        'best_brain_only': brain.iloc[0].to_dict() if len(brain) else None,
        'note': 'the maximum of a large search; the null is what makes it '
                'interpretable. Behaviour is near-circular for this contrast.',
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
