import numpy as np


def chatterjee_xicoef(X: list) -> np.ndarray:
  """Fast Chatterjee Xi correlation coefficient.

  References:
    https://github.com/czbiohub/xicor/issues/17
    R implementation: https://github.com/cran/XICOR/blob/master/R/calculateXI.R

  Parameters
  ----------
  X : list[np.ndarray]
      list of size n_subjects, each item numpy array of size (n_regions, n_timepoints)

  Returns
  -------
  np.ndarray
      Xi matrix of size n_subjects * n_regions * n_regions
  """

  X_xi = []
  for X_sub in X:

    # X is in scikit format, replace region and time axes
    X_sub = X_sub.transpose(1, 0)

    subj_te = np.zeros((X_sub.shape[0], X_sub.shape[0]))

    n = X_sub.shape[1]

    for i, source in enumerate(X_sub):
      for j, target in enumerate(X_sub):
        target = target[np.argsort(source, kind='quicksort')]
        _, inverse, counts = np.unique(target, return_inverse=True, return_counts=True)
        right = np.cumsum(counts)[inverse]
        left = np.cumsum(np.flip(counts))[(counts.size - 1) - inverse]
        coef = 1. - 0.5 * np.abs(np.diff(right)).sum() / np.mean(left * (n - left))
        subj_te[i, j] = coef

    X_xi.append(subj_te)

  X_xi = np.stack(X_xi)

  return X_xi
