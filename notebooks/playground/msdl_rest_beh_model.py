# %%

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import nilearn
from nilearn import plotting
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import sklearn

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR


# %% 1. load data/atlas and create a masker

# TODO use pybids to load data
subjects = [
    'AVGP01', 'AVGP02', 'AVGP03', 'AVGP04', 'AVGP05', 'AVGP08', 'AVGP10',
    'AVGP12NEW', 'AVGP13', 'AVGP13NEW', 'AVGP14NEW', 'AVGP16NEW',
    'AVGP17NEW', 'AVGP18', 'AVGP20']

data = [
    str(Path(f'data/julia2018/bids/sub-{sub}/ses-rest/func/'
             f'sub-{sub}_ses-rest_task-rest_bold.nii.gz'))
    for sub in subjects
]

# generate some fake behavioral features
beh_features = np.asarray([np.random.randn() * 100. for s in subjects])

# MSDL atlas
msdl = nilearn.datasets.fetch_atlas_msdl()
msdl_coords = plotting.find_probabilistic_atlas_cut_coords(msdl.maps)
plotting.plot_prob_atlas(msdl.maps)

# create masker
masker = NiftiMapsMasker(msdl.maps, standardize=True,
                         memory='tmp/nilearn_cache')


# %% 2. extract ROI time series
timeseries = []

for d in data:
    ts = masker.fit_transform(d)
    # ts = masker.fit_transform(d, confounds=data.confounds)
    timeseries.append(ts)

timeseries = np.asarray(timeseries)

plt.imshow(timeseries[0].T)
plt.title('subject0 time series - {} regions'.format(timeseries[0].shape[1]))
plt.show()

# %% 3. calculate connectivity measures

measures = ['correlation', 'partial correlation']  # , 'tangent']

conn_matrices = {}
conn_measures = {}

for m in measures:
    conn_measures[m] = ConnectivityMeasure(kind=m)
    conn_matrices[m] = conn_measures[m].fit_transform(timeseries)[0]


# visualize mean partial correlation

conn_matrix = conn_measures['partial correlation'].mean_

np.fill_diagonal(conn_matrix, 0)  # improves visualization!

plotting.plot_matrix(conn_matrix, labels=msdl.labels,
                     vmax=0.8, vmin=-0.8,
                     colorbar=True)

plotting.plot_connectome(conn_matrix, msdl_coords,
                         edge_threshold="90%",
                         colorbar=True)

plt.show()

# %% 4. model fake behavioral data using resting state connectivities

cv = KFold(n_splits=3, shuffle=True)

mses = []

for train, test in cv.split(timeseries, beh_features):
    conn_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
    connectomes = conn_measure.fit_transform(timeseries[train])
    model = SVR(kernel='linear').fit(connectomes, beh_scores[train])
    # make predictions for the left-out test subjects
    y_true = beh_features[test]
    y_pred = model.predict(conn_measure.transform(timeseries[test]))
    # store the accuracy for this cross-validation fold
    mse = mean_squared_error(y_true, y_pred)
    mses.append(mse)


print('CV-MSE (3 folds):', np.mean(mses))
