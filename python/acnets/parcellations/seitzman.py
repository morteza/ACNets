import pandas as pd
from sklearn.utils import Bunch
from nilearn import datasets, maskers


def load_masker(atlas_name: str, mask_img=None, t_r=3.0):

    if atlas_name != 'seitzman2018':
        raise ValueError('`atlas_name` must be seitzman2018')

    atlas = datasets.fetch_coords_seitzman_2018(
        ordered_regions=True,
        legacy_format=False)

    atlas.pop('description', None)
    atlas_coords = atlas['rois'].values

    masker = maskers.NiftiSpheresMasker(
        seeds=atlas_coords,
        smoothing_fwhm=6,
        # TODO isn't it an atlas with varying ROI radius (either 4 or 5)?
        radius=4,
        allow_overlap=True,
        #   detrend=True,
        #   standardize=True,
        #   low_pass=0.08,
        #   high_pass=0.009,
        t_r=t_r,
        verbose=0)

    atlas = pd.concat([
        pd.DataFrame(v, columns=[k] if k!='rois' else v.columns) for k, v in atlas.items()
    ], axis=1)
    atlas.rename(
        columns={'networks': 'network', 'regions': 'anatomical_region'},
        inplace=True)
    atlas.index.name = 'region'

    return masker, atlas
