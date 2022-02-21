import numpy as np
import sys

def transfer_entropy(X) -> np.ndarray:
  """Calculate transfer entropy between all pairs of nodes in X.

  Parameters
  ----------
  X : numpy.ndarray
      time series of shape (n_subjects, n_regions, n_timepoints)

  Returns
  -------
  np.ndarray
      Adjacency matrix of size (n_subjects, n_regions, n_regions)
  """

  # pyinform is required for the transfer entropy calculation
  import pyinform

  X_te = []
  for X_sub in X:

    # X is in scikit format, replace region and time axes
    X_sub = X_sub.transpose(1, 0)

    X_sub -= X_sub.mean(axis=1, keepdims=True)
    X_sub /= X_sub.std(axis=1, keepdims=True) + sys.float_info.epsilon
    X_sub, *_ = pyinform.utils.bin_series(X_sub, b=101)

    subj_te = np.zeros((X_sub.shape[0], X_sub.shape[0]))

    for i, source in enumerate(X_sub):
      for j, target in enumerate(X_sub):
        subj_te[i, j] = pyinform.transfer_entropy(source, target, k=1)

    X_te.append(subj_te)

  X_te = np.stack(X_te)

  return X_te
