"""What is the theoretical ceiling on decoding accuracy for this design?

Not "what did our classifiers reach" but "what could any classifier reach, ever,
on data of this kind". Four ceilings apply, they are independent, and the
binding one is whichever is lowest. Each is computed rather than asserted.

**1. The Bayes ceiling.** For two Gaussians with a shared covariance, no
classifier with unlimited training data can beat Phi(D/2), where D is the
Mahalanobis distance between the groups. This is a property of the data, not of
the model. Estimated with the Lachenbruch bias correction, because the raw
sample D is wildly optimistic at p close to n.

**2. The reliability ceiling, which is the one nobody computes.** Functional
connectivity is measured with error: mean edge-level test-retest reliability is
ICC = 0.29 (Noble, Scheinost and Constable 2019). A noisy measure of a latent
quantity attenuates the effect by sqrt(ICC). So even if action gaming produced a
*perfect* separation in the latent connectivity of the brain, the separation
visible in a 6-minute scan is bounded. This ceiling binds hardest and is the
reason the field's accuracies cluster where they do.

**3. The finite-sample ceiling.** A classifier trained on n participants does not
reach the Bayes rate; the excess error falls roughly as p/n. Measured here
empirically by a learning curve -- training on subsamples and testing on the
rest -- which also answers the practical question: would more participants help,
or is the curve already flat?

**4. The estimation ceiling.** At n = 32 a measured accuracy has a Wilson
interval about 0.32 wide. Even a true 0.90 could not be distinguished from 0.74
with this sample. This bounds not what is achievable but what is *knowable*.

    uv run python scripts/run_ceiling_theory.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure
from acnets.multiverse.runner import edge_names, vectorize
from estimate_ceiling import bayes_accuracy, mahalanobis
from run_agreement import aggregate, group_labels, load_atlas

OUT = Path('outputs/ceiling')

# Noble, Scheinost & Constable (2019), meta-analysis of FC test-retest:
# mean edge-level ICC 0.29 [0.23, 0.36]; strong within-network cortical edges
# reach roughly 0.5, which is used as the optimistic variant below.
ICC_MEAN, ICC_LOW, ICC_HIGH = 0.29, 0.23, 0.36
ICC_BEST_EDGES = 0.50


def reliability_ceiling(latent_d: float, icc: float) -> float:
    """Best achievable accuracy when the latent effect is measured with error.

    A measure with reliability `icc` attenuates a latent standardised difference
    by sqrt(icc). The ceiling is then the Bayes accuracy of the *attenuated*
    effect, because that is all the measurement contains.
    """
    return float(stats.norm.cdf(latent_d * np.sqrt(icc) / 2))


def learning_curve(x, y, sizes, n_draws=200, seed=0):
    """Accuracy against training-set size, testing on the held-out remainder.

    Regularised LDA throughout: it is the estimator that degrades most
    gracefully at p close to n, so the curve reflects the information available
    rather than a particular model's fragility.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for size in sizes:
        per_class = size // 2
        scores = []
        for _ in range(n_draws):
            train = np.concatenate([
                rng.choice(np.flatnonzero(y == 0), per_class, replace=False),
                rng.choice(np.flatnonzero(y == 1), per_class, replace=False)])
            test = np.setdiff1d(np.arange(len(y)), train)
            if len(np.unique(y[test])) < 2:
                continue
            scaler = StandardScaler().fit(x[train])
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            model.fit(scaler.transform(x[train]), y[train])
            scores.append((model.predict(scaler.transform(x[test])) == y[test]).mean())
        out[size] = (float(np.mean(scores)), float(np.std(scores)))
    return out


def required_n(observed_d: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Participants per group needed to detect `observed_d` at the given power."""
    if observed_d <= 0:
        return 10 ** 9
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return int(np.ceil(2 * ((z_alpha + z_beta) / observed_d) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, args.seed)
    measure = ExtraConnectivityMeasure(kind='partial correlation', vectorize=False)
    measure.fit(timeseries['network'])
    x = vectorize(measure.transform(timeseries['network']), False)
    names = edge_names(nodes['network'], False)
    payload: dict = {}

    print('=' * 72)
    print('1. BAYES CEILING -- the best any classifier could do with infinite data')
    print('=' * 72)
    naive, corrected = mahalanobis(x, y)
    print(f'  all 21 edges:  D^2 naive {naive:.2f} -> bias-corrected {corrected:.2f}')
    print(f'  ceiling: {bayes_accuracy(corrected):.3f}')
    print('  (a negative corrected D^2 means no separation is detectable at all)')
    payload['bayes'] = {'d2_naive': naive, 'd2_corrected': corrected,
                        'ceiling': bayes_accuracy(corrected)}

    con = names.index('CON ↔ SMN')
    single = x[:, [con]]
    d_observed = float(abs(single[y == 0].mean() - single[y == 1].mean())
                       / np.sqrt(((single[y == 0].std(ddof=1) ** 2)
                                  + (single[y == 1].std(ddof=1) ** 2)) / 2))
    print(f'\n  strongest single edge (CON-SMN): observed Cohen d = {d_observed:.2f}')
    print(f'  its Bayes ceiling: {stats.norm.cdf(d_observed / 2):.3f}')
    payload['best_edge'] = {'edge': 'CON ↔ SMN', 'cohens_d': d_observed,
                            'ceiling': float(stats.norm.cdf(d_observed / 2))}

    print()
    print('=' * 72)
    print('2. RELIABILITY CEILING -- the binding one')
    print('=' * 72)
    print(f'  edge-level test-retest ICC = {ICC_MEAN} [{ICC_LOW}, {ICC_HIGH}]')
    print('  (Noble, Scheinost & Constable 2019)\n')
    print(f'  {"latent effect":>16s}   {"ICC=0.29":>9s} {"ICC=0.50":>9s} '
          f'{"ICC=1.00":>9s}   interpretation')
    rows = {}
    for latent_d, label in [(0.5, 'small'), (0.8, 'large'), (1.2, 'very large'),
                            (2.0, 'enormous'), (3.0, 'implausible'),
                            (5.0, 'near-perfect')]:
        a = reliability_ceiling(latent_d, ICC_MEAN)
        b = reliability_ceiling(latent_d, ICC_BEST_EDGES)
        c = reliability_ceiling(latent_d, 1.0)
        rows[latent_d] = {'icc_029': a, 'icc_050': b, 'perfect': c}
        print(f'  d = {latent_d:<12.1f}   {a:9.3f} {b:9.3f} {c:9.3f}   {label}')
    payload['reliability'] = {'icc': ICC_MEAN, 'by_latent_d': rows}

    # what latent effect would be needed for 80% and 90%, at the measured ICC
    for target in (0.80, 0.90):
        needed = 2 * stats.norm.ppf(target) / np.sqrt(ICC_MEAN)
        print(f'\n  to reach {target:.0%} you would need a latent effect of '
              f'd = {needed:.1f}')
        payload.setdefault('latent_d_required', {})[str(target)] = float(needed)
    print('  For scale: d = 5 means the group distributions barely overlap. No')
    print('  individual-differences effect in cognitive neuroscience is that big.')

    print()
    print('=' * 72)
    print('3. FINITE-SAMPLE CEILING -- does more data help?')
    print('=' * 72)
    sizes = [8, 12, 16, 20, 24, 28]
    curve = learning_curve(x, y, sizes, seed=args.seed)
    print(f'  {"train n":>8s}  {"accuracy":>9s}  {"sd":>6s}')
    for size, (mean, sd) in curve.items():
        print(f'  {size:8d}  {mean:9.3f}  {sd:6.3f}')
    first, last = curve[sizes[0]][0], curve[sizes[-1]][0]
    slope = (last - first) / (sizes[-1] - sizes[0])
    print(f'\n  slope over the observed range: {slope:+.4f} accuracy per participant')
    if slope > 0.002:
        print(f'  extrapolating naively, reaching 0.80 would need about '
              f'{int(sizes[-1] + (0.80 - last) / slope)} training participants')
    else:
        print('  the curve is flat: within this range, more participants of this')
        print('  kind do not buy accuracy, which is what a null signal looks like')
    payload['learning_curve'] = {str(k): v for k, v in curve.items()}
    payload['learning_curve_slope'] = float(slope)

    print()
    print('=' * 72)
    print('4. ESTIMATION CEILING -- what is knowable at this n')
    print('=' * 72)
    for accuracy in (0.70, 0.80, 0.90):
        low = stats.norm.ppf(0.025, accuracy,
                             np.sqrt(accuracy * (1 - accuracy) / 32))
        high = stats.norm.ppf(0.975, accuracy,
                              np.sqrt(accuracy * (1 - accuracy) / 32))
        print(f'  a true accuracy of {accuracy:.2f} at n = 32 is measured as '
              f'[{low:.2f}, {high:.2f}]')
    print('\n  the intervals overlap heavily: at n = 32, 0.70 and 0.90 are not')
    print('  reliably distinguishable from one another')

    print()
    print('=' * 72)
    print('BINDING CONSTRAINT')
    print('=' * 72)
    print(f'  n per group for 80% power at the observed CON-SMN effect '
          f'(d = {d_observed:.2f}): {required_n(d_observed)}')
    print(f'  n per group if the true effect is half that: '
          f'{required_n(d_observed / 2)}')
    print('\n  This study has 16 per group.')
    payload['required_n_per_group'] = {
        'at_observed_d': required_n(d_observed),
        'at_half_observed_d': required_n(d_observed / 2)}

    (OUT / 'theory.json').write_text(json.dumps(payload, indent=2, default=float))
    print(f'\nwritten to {OUT}/theory.json')


if __name__ == '__main__':
    main()
