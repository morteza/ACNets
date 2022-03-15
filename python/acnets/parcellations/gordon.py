from pathlib import Path
import pandas as pd
from sklearn.utils import Bunch
from nilearn import maskers
import re


def fetch_atlas(atlas_version_name='gordon2014',
                resolution_mm=2,
                data_dir='data/'):
    """Fetch Gordon2022 atlas from the data directory.

    Parameters
    ----------
    atlas_version_name : str, optional
        The name of the atlas version. Defaults to 'gordon2014'.
    resolution_mm : int, optional
        The resolution of the atlas in millimeters. Defaults to 2.
    data_dir : str, optional
        The data directory where `gordon2014/` resides. Defaults to 'data/'.

    Returns
    -------
    Bunch
        With `labels` and `maps` attributes.
    """
    data_dir = Path(data_dir) / atlas_version_name

    labels = pd.read_csv(data_dir / 'Parcels.csv')
    
    labels.rename(columns={
        'ParcelID': 'region',
        'Hem': 'hemisphere',
        'Community': 'network',
        'Surface area (mm2)': 'surface_area_mm2',
    }, inplace=True)

    # extract MNI coords
    coords = labels['Centroid (MNI)'].apply(lambda x: pd.Series(x.split(' ')))
    labels[['x', 'y', 'z']] = coords
    labels.drop(columns=['Centroid (MNI)'], inplace=True)
    labels.set_index('region', inplace=True)

    atlas_maps = f'{data_dir}/Parcels_MNI_{str(resolution_mm)*3}.nii'

    return Bunch(labels=labels, maps=atlas_maps)


def load_masker(atlas_name, mask_img, t_r=3.0):

    atlas_version_name, resolution_mm = re.findall('(gordon\\d{4})_(\\d+)mm', atlas_name)[0]
    resolution_mm = int(resolution_mm)

    atlas = fetch_atlas(atlas_version_name, resolution_mm)

    masker = maskers.NiftiLabelsMasker(
        atlas.maps,
        labels=['0'] + [str(lbl) for lbl in atlas.labels.index],
        mask_img=mask_img,
        detrend=True,
        standardize=True,
        t_r=t_r,
        verbose=0)

    # TODO return Bunch instead
    return masker, atlas.labels
