"""Murphy ch.19 methods that actually apply here: augmentation and weak supervision.

Most of "Learning with Fewer Labeled Examples" does not apply to this design, and
the reason is worth stating once because it is the same reason each time:

| method | why not |
|---|---|
| semi-supervised (19.3) | needs an *unlabelled pool from the target
  distribution*. LEMON and ABIDE are unlabelled but are not drawn from a
  gamer/non-gamer population, so a pseudo-label for them means nothing |
| active learning (19.4) | needs the ability to acquire labels. Group
  membership is known for all 32 participants; there is nothing to query |
| meta-learning (19.5) | needs a *distribution of related tasks* to meta-learn
  across. There is one task |
| few-shot (19.6) | same. 16 per class is the whole dataset, not a support set
  drawn from many classes |
| transfer (19.2) | already tested four ways: fine-tuning 0.477, diffusion
  density at chance, tangent reference 1.6% of variance, reliability weighting
  +0.031 |
| domain adaptation (19.2.5) | the right tool for gong2015 <-> Julia2018, and
  blocked on the skull-stripping defect in section 16 |

Two do apply, and neither has been tried on the classifier.

**Data augmentation (19.1).** A resting run can be cut into overlapping windows,
each giving its own connectivity estimate. Murphy's justification (19.1.2) is
that augmentation approximates an expectation over a transformation group -- it
is a variance reducer that enforces an invariance. The invariance here is real
and worth enforcing: a participant's group does not depend on which 90 seconds
of their scan you look at. And it targets the binding constraint directly, since
within-run estimation noise *is* what limits reliability (section 13).

**Weak supervision / multiple-instance learning (19.7).** The structure maps
exactly: each participant is a **bag**, their windows are **instances**, and the
label exists only at bag level. Rather than averaging windows into one matrix and
throwing the variation away, MIL pools instance-level evidence -- and mean versus
max pooling encode different hypotheses. Mean says every window carries the
signal; max says only some do, which is a real possibility for a state that comes
and goes across a resting run.

**The one thing that must not go wrong**: windows from one participant are not
independent. Every split is at *participant* level, so no window of a held-out
participant is ever seen in training. Without that, this whole approach is an
elaborate way to leak.

    uv run python scripts/run_weak_supervision.py --n-permutations 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

OUT = Path('outputs/weak_supervision')
STATISTIC = 'precision'
# discovered in this dataset (manuscript 3.1), not declared in advance
DISCOVERED_NETWORKS = ('FPN', 'CON', 'SMN')


def windows(timeseries: np.ndarray, length: int, stride: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """(instances, bag_index). Overlapping windows, one bag per participant."""
    n_subjects, n_timepoints, _ = timeseries.shape
    starts = list(range(0, n_timepoints - length + 1, stride))
    instances, bags = [], []
    for subject in range(n_subjects):
        for start in starts:
            instances.append(timeseries[subject, start:start + length, :])
            bags.append(subject)
    return np.stack(instances), np.array(bags)


def restrict(x: np.ndarray, names: list[str]) -> np.ndarray:
    """The section 15 restriction: the three networks discovered in 3.1."""
    pairs = [tuple(n.split(' ↔ ')) for n in names]
    keep = [i for i, (a, b) in enumerate(pairs)
            if a in DISCOVERED_NETWORKS and b in DISCOVERED_NETWORKS]
    return x[:, keep]


def model():
    return LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=OAS())


def baseline(x: np.ndarray, y: np.ndarray) -> float:
    """One connectivity matrix per participant, no augmentation."""
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneOut().split(y):
        scaler = StandardScaler().fit(x[train])
        fitted = model().fit(scaler.transform(x[train]), y[train])
        predictions[test] = fitted.predict(scaler.transform(x[test]))
    return float((predictions == y).mean())


def mil(instances: np.ndarray, bags: np.ndarray, y: np.ndarray,
        pooling: str) -> float:
    """Train on windows, predict per participant by pooling their windows.

    Splitting is at participant level: every window of the held-out participant
    is in the test set and none of them was seen in training.
    """
    predictions = np.empty(len(y), dtype=int)
    for train_subjects, test_subjects in LeaveOneOut().split(y):
        train_mask = np.isin(bags, train_subjects)
        test_mask = np.isin(bags, test_subjects)

        scaler = StandardScaler().fit(instances[train_mask])
        # every window of a participant inherits that participant's label --
        # this is the weak-supervision step, and it is what multiplies the
        # training set without inventing any label information
        fitted = model().fit(scaler.transform(instances[train_mask]),
                             y[bags[train_mask]])

        scores = fitted.decision_function(scaler.transform(instances[test_mask]))
        scores = np.atleast_1d(scores)
        if pooling == 'mean':
            pooled = scores.mean()
        elif pooling == 'max':                 # the most confident window wins
            pooled = scores[np.argmax(np.abs(scores))]
        elif pooling == 'vote':
            pooled = np.sign((scores > 0).mean() - 0.5)
        else:
            raise ValueError(pooling)
        predictions[test_subjects] = int(pooled > 0)
    return float((predictions == y).mean())


def build(seed: int, length: int, stride: int):
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, seed)
    names = edge_names(nodes['network'], False)

    full = restrict(connectivity(timeseries['network'], STATISTIC), names)
    instance_ts, bags = windows(timeseries['network'], length, stride)
    instances = restrict(connectivity(instance_ts, STATISTIC), names)
    return full, instances, bags, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=60,
                        help='window length in TRs (TR = 3 s)')
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--n-permutations', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    full, instances, bags, y = build(args.seed, args.length, args.stride)
    per_bag = len(instances) // len(y)
    print(f'{len(y)} participants (bags), {len(instances)} windows '
          f'({per_bag} per participant) of {args.length} TRs')
    print(f'features: {full.shape[1]} edges over {"+".join(DISCOVERED_NETWORKS)}, '
          f'statistic {STATISTIC}\n')

    def evaluate(labels: np.ndarray) -> dict[str, float]:
        return {
            'baseline/whole-run': baseline(full, labels),
            'mil/mean-pool': mil(instances, bags, labels, 'mean'),
            'mil/max-pool': mil(instances, bags, labels, 'max'),
            'mil/vote': mil(instances, bags, labels, 'vote'),
        }

    observed = evaluate(y)
    for cell, value in sorted(observed.items(), key=lambda kv: -kv[1]):
        interval = accuracy_interval(value * len(y), len(y))
        print(f'  {cell:24s} {value:.3f}   [{interval.low:.2f}, {interval.high:.2f}]')

    started = time.perf_counter()
    print(f'\nfamily-wise null over {len(observed)} cells, '
          f'B = {args.n_permutations}:')

    def one(i: int) -> list[float]:
        rng = np.random.default_rng(args.seed + 1 + i)
        # permute at participant level: windows inherit their bag's label, so
        # permuting instances directly would destroy the bag structure and
        # produce an easier null than the observed analysis faces
        scores = evaluate(rng.permutation(y))
        return [scores[k] for k in observed]

    null = np.array(Parallel(n_jobs=args.n_jobs, verbose=1)(
        delayed(one)(i) for i in range(args.n_permutations)))
    print(f'  {time.perf_counter() - started:.0f}s')

    result = max_stat_permutation(observed, null)
    print(f'\n  observed max        {result.observed_max:.3f}')
    print(f'  null max, median    {np.median(result.null_max):.3f}')
    print(f'  null max, 95th pct  {result.critical_value:.3f}')
    print(f'  p                   {result.p_value:.4f}')

    best = max(observed, key=observed.get)
    gain = observed[best] - observed['baseline/whole-run']
    print(f'\n  augmentation/MIL gain over the whole-run baseline: {gain:+.3f}')

    (OUT / 'weak_supervision.json').write_text(json.dumps({
        'statistic': STATISTIC, 'networks': list(DISCOVERED_NETWORKS),
        'window_length': args.length, 'stride': args.stride,
        'windows_per_participant': per_bag,
        'observed': observed, 'gain_over_baseline': gain,
        **result.summary(),
    }, indent=2, default=float))
    pd.DataFrame([{'cell': k, 'accuracy': v} for k, v in observed.items()]).to_csv(
        OUT / 'weak_supervision.csv', index=False)
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
