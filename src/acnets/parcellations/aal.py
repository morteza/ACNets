import pandas as pd
from nilearn import maskers


def load_masker(
        atlas_name='aal',
        t_r: float = 3.0) -> tuple[maskers.NiftiLabelsMasker, pd.DataFrame]:
    # TODO receive data folder as parameter

    labels = pd.read_csv('data/atlases/AAL3v1.csv')
    labels.dropna(subset=['index'], inplace=True)
    labels.set_index('region', inplace=True)

    masker = maskers.NiftiLabelsMasker(
        'data/atlases/AAL3v1.nii.gz',
        standardize='zscore_sample',
        # nilearn 0.14 narrowed this to bool; True is what 'zscore_sample' meant
        standardize_confounds=True,
        resampling_target='labels',
        # detrend=True,
        verbose=0)

    return masker, labels
