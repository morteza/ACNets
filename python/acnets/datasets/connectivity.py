from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd


__supported_parcellations = [
    'dosenbach2007', 'dosenbach2010', 'difumo_64_2', 'difumo_128_2', 'difumo_1024_2',]

__supported_kinds = [
    'tangent', 'precision', 'correlation', 'covariance', 'partial_correlation',
    'chatterjee', 'transfer_entropy']


def __get_feature_name(
    feature_info,
    sep=' \N{left right arrow} '
):

  names = feature_info.index.to_series().apply(
      lambda f1: f1 if f1 == feature_info.name else f'{f1}{sep}{feature_info.name}')

  return names


def load_connectivity(
    dataset='julia2018_resting',
    parcellation='dosenbach2007',
    kind='tangent',
    discard_diagonal=False,
    discard_invalid_subjects=False,
    vectorize=False,
    only_diagonal=False,
    return_y=None,
    return_feature_names=True,
    filename=None,
    **kwargs
):

  if parcellation not in __supported_parcellations:
    raise ValueError('Invalid parcellation atlas: {}'.format(parcellation))

  if kind not in __supported_kinds:
    raise ValueError('Invalid connectivity kind: {}'.format(kind))

  # NOTE these are currently hidden kwargs because of irrelevance to the loading
  threshold = kwargs.get('binarization_threshold', None)
  shuffle = kwargs.get('shuffle', False)  # noqa
  discard_cerebellum = kwargs.get('discard_cerebellum', False)

  filename = filename or (Path('data') / dataset / f'connectivity_{parcellation}.nc')

  ds = xr.open_dataset(filename)

  _conn = ds[f'{kind}_connectivity']
  _conn.coords['group'] = ds.group
  _conn['inverse_efficiency_score_ms'] = ds['inverse_efficiency_score_ms']

  if 'difumo_names' in ds.coords:
    _conn.coords['region'] = ds.coords['difumo_names'].values

  regions = ds.coords['region'].values

  if only_diagonal:
    feature_names = regions
    X = np.array([np.diag(subj_conn) for subj_conn in _conn.values])

  else:
    triu_k = 1 if discard_diagonal else 0

    feature_names = pd.DataFrame(np.empty((len(regions), len(regions))),
                                 index=regions, columns=regions)
    feature_names = feature_names.apply(__get_feature_name)

    X = _conn.values

  if threshold is not None:
    print('Binarizing connectivity matrix... ', end='')

    X_threshold = (
      np.median(X, axis=1, keepdims=True) + threshold * np.std(X, axis=1, keepdims=True))

    X = np.where(np.abs(X) >= X_threshold, X, 0)
    print('done!')

  if discard_invalid_subjects:
    behavioral_scores = ds['inverse_efficiency_score_ms'].values
    subj_labels = xr.concat([ds['subject'], ds['subject'] + 'NEW'], dim='subject')
    invalid_subjects = subj_labels.to_series().duplicated(keep='first')[32:]
    invalid_subjects = invalid_subjects | np.isnan(behavioral_scores)
    invalid_subjects = np.isnan(behavioral_scores)
    X = X[~invalid_subjects]

  if discard_cerebellum:
    raise NotImplementedError('Cerebellum discarding not implemented yet')

  if vectorize:
    X = np.array([
        subj_conn[np.triu_indices_from(subj_conn, k=triu_k)] for subj_conn in X])
    feature_names = feature_names.values[
        np.triu_indices_from(feature_names.values, k=triu_k)]

  if return_y:
    y = ds['group'].values
    if discard_invalid_subjects:
      y = y[~invalid_subjects]

  if shuffle:
    raise NotImplementedError('Shuffling is not implemented yet')

  if return_y and return_feature_names:
    return X, y, feature_names
  elif return_feature_names:
    return X, feature_names
  else:
    return X
