from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd


__supported_parcellations = [
    'dosenbach2007', 'dosenbach2010', 'difumo_64_2mm', 'difumo_128_2mm', 'difumo_1024_2mm',]

__supported_kinds = [
    'tangent', 'precision', 'correlation', 'covariance', 'partial_correlation',
    'chatterjee', 'transfer_entropy']


def __get_feature_name(feature_info,
                       sep=' \N{left right arrow} '):

  names = feature_info.index.to_series().apply(
      lambda f1: f1 if f1 == feature_info.name else f'{f1}{sep}{feature_info.name}')

  return names


def get_networks(features, parcellation):
  from nilearn.datasets import fetch_coords_dosenbach_2010

  if parcellation.lower() == 'dosenbach2010':
    coords = fetch_coords_dosenbach_2010(legacy_format=False)
    labels = pd.concat([
        pd.Series(coords['labels']),
        coords['networks'].reset_index(drop=True)], axis=1)
    labels.rename({0: 'region'}, inplace=True)
    labels.set_index(0, inplace=True)
  elif parcellation.lower() == 'dosenbach2007':
    labels = pd.read_csv('data/dosenbach2007/ROIS.csv', index_col=0)
    labels = labels[['network']]

  networks = []

  for f in features:
    if '↔' in f:
      src, tgt = f.split(' ↔ ')
      src_network = labels.loc[src, 'network']
      tgt_network = labels.loc[tgt, 'network']
      if src_network == tgt_network:
        network = labels.loc[src, 'network']
      else:
        network = f'{src_network} ↔ {tgt_network}'
    else:
      network = labels.loc[f, 'network']

    networks.append((network, f))

  return networks


def load_julia2018_connectivity(
    dataset='julia2018_resting',
    parcellation='dosenbach2007',
    kind='tangent',
    discard_invalid_subjects=False,
    vectorize=False,
    binarize=False,
    only_diagonal=False,
    discard_diagonal=False,
    return_y=None,
    filename=None,
    **kwargs
):

  if parcellation not in __supported_parcellations:
    raise ValueError('Invalid parcellation atlas: {}'.format(parcellation))

  if kind not in __supported_kinds:
    raise ValueError('Invalid connectivity kind: {}'.format(kind))

  # NOTE these are currently hidden kwargs because of irrelevance to the loading
  shuffle = kwargs.get('shuffle', False)  # noqa

  filename = filename or (Path('data') / dataset / f'connectivity_{parcellation}.nc')

  ds = xr.open_dataset(filename)

  _conn = ds[f'{kind}_connectivity']
  _conn.coords['group'] = ds.group
  _conn['inverse_efficiency_score_ms'] = ds['inverse_efficiency_score_ms']

  if 'difumo_names' in ds.coords:
    _conn.coords['region'] = ds.coords['difumo_names'].values

  regions = ds.coords['region'].values

  if only_diagonal:
    diag_conn = np.array([np.diag(subj_conn) for subj_conn in _conn.values])
    X = pd.DataFrame(diag_conn, index=ds.coords['subject'], columns=regions)
  else:
    feature_names = pd.DataFrame(np.empty((len(regions), len(regions))),
                                 index=regions, columns=regions)
    feature_names = feature_names.apply(__get_feature_name)
    X = _conn.values

    if binarize:  # binarize
      X_bin = []
      from sklearn.preprocessing import Binarizer
      for X_subj in X:
        threshold = np.median(X_subj) + float(binarize) * np.std(X_subj)
        X_subj_bin = Binarizer(threshold=threshold).transform(np.abs(X_subj))
        X_bin.append(X_subj_bin)
      X = np.array(X_bin)

  if discard_invalid_subjects:
    behavioral_scores = ds['inverse_efficiency_score_ms'].values
    subj_labels = xr.concat([ds['subject'], ds['subject'] + 'NEW'], dim='subject')
    invalid_subjects = subj_labels.to_series().duplicated(keep='first')[32:]
    invalid_subjects = invalid_subjects | np.isnan(behavioral_scores)
    invalid_subjects = np.isnan(behavioral_scores)
    print(invalid_subjects)
    X = X[~invalid_subjects]

  if vectorize and not only_diagonal:
    triu_k = 1 if discard_diagonal else 0
    vec_conns = np.array(
        [subj_conn[np.triu_indices_from(subj_conn, k=triu_k)]
         for subj_conn in X])

    vec_feature_names = feature_names.values[
        np.triu_indices_from(feature_names.values, k=triu_k)]
    X = pd.DataFrame(vec_conns, index=ds.coords['subject'], columns=vec_feature_names)

  if return_y:
    y = ds['group'].values
    if discard_invalid_subjects:
      y = y[~invalid_subjects]

  X.columns = pd.MultiIndex.from_tuples(get_networks(X.columns, parcellation))

  if shuffle:
    raise NotImplementedError('Shuffling is not implemented yet')

  if return_y:
    return X, y
  else:
    return X
