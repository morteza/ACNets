# %%

import numpy as np

import matplotlib.pyplot as plt

from nilearn import datasets, plotting
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_label_regions

from pathlib import Path

# %% 1. load resting bold data
data_dir = Path('data/julia2018/sub-AVGP01/ses-rest/func/')
data = str(data_dir / 'sub-AVGP01_ses-rest_task-rest_bold.nii.gz')

# EXAMPLE DATA: data_filename = datasets.fetch_development_fmri(n_subjects=1).func[0]

# %% 2. load atlas and init masker

# datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas = yeo_2011.thick_17

region_labels = connected_label_regions(atlas)

coords = plotting.find_parcellation_cut_coords(atlas)

plotting.plot_roi(atlas)

masker = NiftiLabelsMasker(atlas, standardize=True,
                           memory='tmp/nilearn_cache')

# %% [3] extract time series for parcells

time_series = masker.fit_transform(data)
# time_series = masker.fit_transform(data_filename,
#                                    confounds=data.confounds)

plt.imshow(time_series.T)

# %% [4] extract connectivity measures

measures = ['correlation', 'partial correlation', 'tangent']


conn_measure = ConnectivityMeasure(kind='correlation')
conn_matrix = conn_measure.fit_transform([time_series])[0]


# %% 5. plot correlations and connectome

np.fill_diagonal(conn_matrix, 0)  # improves visualization!

plotting.plot_matrix(conn_matrix,
                     labels=masker.labels_,
                     vmax=0.8, vmin=-0.8,
                     colorbar=True)

plotting.plot_connectome(conn_matrix, coords,
                         edge_threshold="80%", colorbar=True)

