"""What accuracy could *any* classifier reach on these 32 participants?

Every search reported in this project asks "can a better pipeline decode the
groups". That question has an upper bound which does not depend on the pipeline
at all, and estimating it says more than another sweep would.

Under the usual assumption behind linear discriminant analysis -- two Gaussians
with a shared covariance -- the best achievable accuracy for *any* classifier
with unlimited training data is

    Bayes accuracy = Phi(D / 2),   D = Mahalanobis distance between the groups.

So estimating D bounds every method at once: no atlas, statistic, architecture
or amount of pretraining can beat Phi(D/2), because that limit is a property of
the data rather than of the model.

**The estimate has to be bias-corrected or it is worthless.** The sample
Mahalanobis distance is badly upward-biased when the feature count approaches
the sample size, which is exactly this regime: 21 features, 32 participants. The
in-sample D would imply a ceiling far above anything cross-validation ever
produces, which is how a small-sample study talks itself into believing a
better model is just around the corner. We use the standard unbiased estimator
(Lachenbruch 1968),

    D2_unbiased = ((N - p - 3) / (N - 2)) * D2_observed - p * N / (n1 * n2),

report it alongside the naive value so the size of the correction is visible,
and bootstrap it for an interval.

    uv run python scripts/estimate_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acnets.connectome import ExtraConnectivityMeasure
from acnets.multiverse.runner import edge_names, vectorize
from run_agreement import aggregate, group_labels, load_atlas

OUT = Path('outputs/ceiling')
STATISTIC = 'partial correlation'


def mahalanobis(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(naive D^2, bias-corrected D^2) between the two groups in `x`."""
    a, b = x[y == 0], x[y == 1]
    n1, n2, p = len(a), len(b), x.shape[1]
    total = n1 + n2

    delta = a.mean(axis=0) - b.mean(axis=0)
    pooled = (((n1 - 1) * np.cov(a, rowvar=False)
               + (n2 - 1) * np.cov(b, rowvar=False)) / (total - 2))
    # ridge for numerical stability when p approaches n; small relative to the
    # feature scale, and applied identically to both the naive and corrected value
    pooled = np.atleast_2d(pooled)
    pooled += np.eye(p) * 1e-6 * np.trace(pooled) / max(p, 1)

    naive = float(delta @ np.linalg.solve(pooled, delta))
    if total - p - 3 <= 0:
        return naive, float('nan')
    corrected = ((total - p - 3) / (total - 2)) * naive - p * total / (n1 * n2)
    return naive, float(corrected)


def bayes_accuracy(d_squared: float) -> float:
    """Phi(D/2), the best accuracy any classifier could reach."""
    if not np.isfinite(d_squared) or d_squared <= 0:
        return 0.5
    return float(stats.norm.cdf(np.sqrt(d_squared) / 2))


def bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
              seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(np.flatnonzero(y == 0), (y == 0).sum(), replace=True),
            rng.choice(np.flatnonzero(y == 1), (y == 1).sum(), replace=True)])
        try:
            values.append(bayes_accuracy(mahalanobis(x[idx], y[idx])[1]))
        except np.linalg.LinAlgError:
            continue
    return (float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_atlas('dosenbach2010')
    y, _ = group_labels(dataset)
    timeseries, nodes = aggregate(dataset, 0)

    measure = ExtraConnectivityMeasure(kind=STATISTIC, vectorize=False)
    measure.fit(timeseries['network'])
    x_all = vectorize(measure.transform(timeseries['network']), False)
    names = edge_names(nodes['network'], False)

    top_two = [i for i, n in enumerate(names) if n in ('CON ↔ SMN', 'FPN ↔ SMN')]

    feature_sets = {
        'all 21 network edges': np.arange(x_all.shape[1]),
        'the 2 attributed edges': np.array(top_two),
        'CON-SMN alone': np.array(top_two[:1]),
    }

    print(f'{len(y)} participants, statistic = {STATISTIC}\n')
    payload = {}
    for label, columns in feature_sets.items():
        x = x_all[:, columns]
        naive, corrected = mahalanobis(x, y)
        ceiling = bayes_accuracy(corrected)
        low, high = bootstrap(x, y)
        payload[label] = {
            'n_features': int(x.shape[1]),
            'd2_naive': naive, 'd2_corrected': corrected,
            'bayes_accuracy': ceiling, 'bayes_ci95': [low, high]}
        print(f'  {label:24s} p = {x.shape[1]:2d}   '
              f'D^2 naive {naive:6.2f} -> corrected {corrected:6.2f}   '
              f'ceiling {ceiling:.3f} [{low:.2f}, {high:.2f}]')

    # what the multiverse demands, for comparison
    critical = json.loads(
        Path('outputs/permutation/permutation.json').read_text())['critical_value_95']
    best = max(v['bayes_accuracy'] for v in payload.values())
    print(f'\n  family-wise critical value to clear   {critical:.3f}')
    print(f'  best estimated ceiling of the data    {best:.3f}')
    print('  => ' + ('the bar is ABOVE the ceiling: no method can clear it here'
                     if best < critical else
                     'the ceiling is nominally above the bar -- but read the '
                     'selection analysis below before believing it'))

    # ---- what selection alone buys, and what it costs ---------------------
    # The ceiling above for "the 2 attributed edges" is optimistic: those two
    # edges are the top of an attribution ranking computed on these same 32
    # participants, so choosing them used the labels indirectly. The honest
    # version puts the choice inside the folds. The gap between the two is the
    # size of the circularity, and it is the number that decides whether
    # "restrict to the best features" is a real strategy or an illusion.
    from scipy.stats import ttest_ind
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    from acnets.multiverse import fixed_splits

    splits = fixed_splits(len(y), 4, 25, 0)

    def fixed_columns(columns):
        folds = []
        for train, test in splits:
            scaler = StandardScaler().fit(x_all[np.ix_(train, columns)])
            model = LinearDiscriminantAnalysis().fit(
                scaler.transform(x_all[np.ix_(train, columns)]), y[train])
            folds.append((model.predict(
                scaler.transform(x_all[np.ix_(test, columns)])) == y[test]).mean())
        return float(np.mean(folds))

    def selected_in_fold(k):
        folds, picks = [], []
        for train, test in splits:
            a = x_all[train][y[train] == 0]
            b = x_all[train][y[train] == 1]
            score = np.nan_to_num(np.abs(ttest_ind(a, b, equal_var=False).statistic))
            columns = np.argsort(score)[::-1][:k]
            picks.append(columns)
            scaler = StandardScaler().fit(x_all[np.ix_(train, columns)])
            model = LinearDiscriminantAnalysis().fit(
                scaler.transform(x_all[np.ix_(train, columns)]), y[train])
            folds.append((model.predict(
                scaler.transform(x_all[np.ix_(test, columns)])) == y[test]).mean())
        return float(np.mean(folds)), picks

    print('\nselecting features on all the data vs inside the folds (LDA):')
    circular = fixed_columns(np.array(top_two))
    honest, picks = selected_in_fold(2)
    con_index = names.index('CON \u2194 SMN')
    recovered = sum(con_index in p for p in picks)
    print(f'  2 edges chosen on all 32 participants   {circular:.3f}  <- circular')
    print(f'  2 edges chosen inside each training fold {honest:.3f}  <- honest')
    print(f'  the gap, i.e. the selection effect       {circular - honest:+.3f}')
    print(f'  CON-SMN recovered by in-fold selection in {recovered}/{len(picks)} folds')

    payload['selection'] = {
        'circular_two_edges': circular,
        'nested_two_edges': honest,
        'selection_effect': circular - honest,
        'con_smn_recovered_folds': int(recovered),
        'n_folds': len(picks),
        'note': 'the circular number is what "just use the best features" looks '
                'like; the nested number is what it is worth'}

    payload['family_wise_critical_value'] = critical
    payload['verdict'] = ('bar above ceiling' if best < critical
                          else 'ceiling above bar')
    (OUT / 'ceiling.json').write_text(json.dumps(payload, indent=2, default=float))
    print(f'\nwritten to {OUT}/')


if __name__ == '__main__':
    main()
