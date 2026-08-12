"""A spatially-displaced negative control for the network-restricted result.

Section 15 restricts connectivity to FPN, CON and SMN and reaches 0.812. Those
three networks were *discovered* in this dataset (manuscript 3.1), not declared
in advance, so the number needs two alternative explanations ruled out, not one.
The first is circularity, which no within-sample control can fix and which
`run_crossmodal_networks.py` addresses by testing the same networks elsewhere.
The second is dimensionality: restriction reduces it, and at n = 32 *any*
reduction helps. This script addresses the second, by restricting somewhere the
gaming literature gives no reason to look.

The cerebellum is that somewhere. It is spatially displaced from the cortical
control-sensorimotor account, and Dosenbach2010 assigns 18 of its 160 regions to
it -- so the control runs on the cached timeseries with no new parcellation. The
manuscript previously recorded this as blocked on DiFuMo-256; it is not.

**Everything here is matched to k = 18 regions.** That is the whole design. An
unmatched comparison -- 18 cerebellar regions against the 86 regions of
FPN+CON+SMN, or against all 160 -- would confound *where* with *how many* and
answer neither question. (It is also intractable: shrinkage LDA on the 12,720
edges of the full atlas is an O(p^3) solve per fold.) So three sets of 18:

**cerebellum** -- the 18 CER regions, 153 edges.
**discovered** -- 18 regions drawn from FPN+CON+SMN, over many draws.
These three are the networks section 3.1 found the classifiers leaning on in
*this* sample; nothing declared them in advance, which is exactly why they need
a size-matched comparison rather than a free pass.
**elsewhere** -- 18 regions drawn from anywhere non-cerebellar, over many draws.

If the cerebellum sits in the middle of *elsewhere*, restriction is buying
dimensionality rather than localisation, and section 15's result is about how
many features it kept, not which. If the discovered set sits above both, the
localisation is doing work.

    uv run python scripts/run_cerebellar_control.py --n-draws 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.multiverse import accuracy_interval
from run_agreement import group_labels, load_atlas
from run_reliability_transfer import connectivity, loocv

OUT = Path('outputs/controls')
STATISTIC = 'precision'
CEREBELLUM = 'CER'
# discovered in this dataset (manuscript 3.1), not declared in advance
DISCOVERED_NETWORKS = ('FPN', 'CON', 'SMN')


def restricted_accuracy(timeseries: np.ndarray, columns, y) -> float:
    """Connectivity estimated *within* the kept regions, then classified.

    The estimate is rebuilt from the retained time courses rather than sliced
    out of the full connectivity matrix. For precision (and any other inverse-
    covariance statistic) those differ: an entry of the full inverse covariance
    conditions on all 160 regions, while the restricted estimate conditions
    only on the subset. The subset estimate is the one that matches what a
    restricted analysis would actually have.
    """
    return loocv(connectivity(timeseries[:, :, list(columns)], STATISTIC), y)


def draw_accuracies(timeseries, pool, k, y, n_draws, seed, n_jobs):
    """Accuracy for `n_draws` random k-subsets of `pool`."""
    rng = np.random.default_rng(seed)
    draws = [rng.choice(pool, size=k, replace=False) for _ in range(n_draws)]
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(restricted_accuracy)(timeseries, columns, y)
        for columns in draws))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-draws', type=int, default=200,
                        help='size-matched random region subsets, per pool')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries = dataset['timeseries'].transpose(
        'subject', 'timepoint', 'region').values
    networks = [str(n) for n in np.asarray(dataset['network'].values)]

    cerebellar = [i for i, n in enumerate(networks) if n == CEREBELLUM]
    discovered = [i for i, n in enumerate(networks)
                  if n in DISCOVERED_NETWORKS]
    elsewhere = [i for i, n in enumerate(networks) if n != CEREBELLUM]
    k = len(cerebellar)
    print(f'{len(networks)} regions: {k} cerebellar, {len(discovered)} in '
          f'{"+".join(DISCOVERED_NETWORKS)}, {len(elsewhere)} non-cerebellar\n')

    observed = restricted_accuracy(timeseries, cerebellar, y)
    interval = accuracy_interval(observed * len(y), len(y))
    print(f'cerebellum ({k} regions, {k * (k - 1) // 2} edges)  {observed:.3f}  '
          f'[{interval.low:.2f}, {interval.high:.2f}]\n')

    pools = {'discovered': discovered, 'elsewhere': elsewhere}
    distributions = {
        name: draw_accuracies(timeseries, pool, k, y, args.n_draws,
                              args.seed, args.n_jobs)
        for name, pool in pools.items()}

    summary = {}
    for name, values in distributions.items():
        percentile = float((values < observed).mean() * 100)
        summary[name] = {
            'median': float(np.median(values)),
            'q95': float(np.quantile(values, 0.95)),
            'max': float(values.max()),
            'cerebellum_percentile': percentile}
        print(f'{args.n_draws} draws of {k} from {name:9s} '
              f'({len(pools[name])} regions): '
              f'median {np.median(values):.3f}, '
              f'95th {np.quantile(values, 0.95):.3f}, '
              f'max {values.max():.3f}')
        print(f'  cerebellum sits at the {percentile:.0f}th percentile\n')

    localised = (np.median(distributions['discovered'])
                 - np.median(distributions['elsewhere']))
    print(f'discovered minus elsewhere, median: {localised:+.3f}  '
          f'-> {"localisation does work" if localised > 0.03 else
              "size, not location, is what restriction buys"}')

    (OUT / 'cerebellar_control.json').write_text(json.dumps({
        'statistic': STATISTIC,
        'k': k,
        'pool_sizes': {'cerebellum': k, 'discovered': len(discovered),
                       'elsewhere': len(elsewhere)},
        'cerebellum_accuracy': observed,
        'n_draws': args.n_draws,
        'size_matched': summary,
        'discovered_minus_elsewhere_median': float(localised),
        'note': ('region-scale, not the network-scale grid of section 15; '
                 'every set is k = 18 regions so that dimensionality is held '
                 'fixed; connectivity is re-estimated within each subset so '
                 'that precision conditions on the subset rather than on all '
                 '160 regions'),
    }, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
