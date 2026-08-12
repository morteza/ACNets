import pandas as pd
from nilearn import datasets, maskers


def load_masker(atlas_name: str, mask_img=None, t_r=3.0):

    if atlas_name != 'seitzman2018':
        raise ValueError('`atlas_name` must be seitzman2018')

    # legacy_format was dropped in nilearn 0.13; its non-legacy shape, which is
    # what the code below assumes, is now the only one returned.
    atlas = datasets.fetch_coords_seitzman_2018(ordered_regions=False)

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


    atlas_labels = pd.concat(
        [pd.DataFrame(v) for _, v in atlas.items()],
        ignore_index=False,
        axis=1)

    atlas_labels.index.name = 'region'

    atlas_labels.columns = ['x', 'y', 'z', 'radius', 'network', 'anatomical_region']

    return masker, atlas_labels
