# %%

import numpy as np

import matplotlib.pyplot as plt

from nilearn import datasets, plotting
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_label_regions

from pathlib import Path

# %% [1] load resting bold data
data_dir = Path('data/julia2018/bids/sub-AVGP01/ses-rest/func/')
data_filename = str(data_dir / 'sub-AVGP01_ses-rest_task-rest_bold.nii.gz')

# EXAMPLE DATA: data_filename = datasets.fetch_development_fmri(n_subjects=1).func[0]

# %% [2] load atlas and prepare ROI masker

# datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_filename = atlas_yeo_2011.thick_17

region_labels = connected_label_regions(atlas_filename)

coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)


plotting.plot_roi(atlas_filename)

masker = NiftiLabelsMasker(labels_img=atlas_filename,
                           memory='nilearn_cache',
                           standardize=True)


# %% [3] extract time series for parcells

time_series = masker.fit_transform(data_filename)
# time_series = masker.fit_transform(data_filename,
#                                    confounds=data.confounds)

plt.imshow(time_series.T)

# %% [4] extract connectivity measures (e.g., correlation)

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
np.fill_diagonal(correlation_matrix, 0)


# %% [5] plot correlations and connectome
# Display the correlation matrix
plotting.plot_matrix(correlation_matrix,
                     labels=masker.labels_,
                     colorbar=True,
                     vmax=0.8,
                     vmin=-0.8)

plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="80%", colorbar=True)

