"""Per-region grey-matter volumes from fmriprep's aparc+aseg segmentations.

The structural modality of the multiview model (plan.md C4), extracted without
any new toolchain: fmriprep already ran FreeSurfer-style segmentation for every
Julia2018 subject, and a volume per labelled region is a voxel count times the
voxel volume. Volumes are read from the native-space segmentation — the
MNI-resampled copy measures the template's anatomy as much as the subject's.

Features are named `fs<label>` by FreeSurfer label id. The multiview model
consumes them as an opaque vector, so no LUT is required; anyone interpreting a
specific label reads it against the standard FreeSurferColorLUT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def regional_volumes(segmentation: str | Path) -> pd.Series:
    """Volume in mm^3 per label in one aparc+aseg image, background excluded."""
    import nibabel as nib

    image = nib.load(str(segmentation))
    voxel_volume = float(np.prod(image.header.get_zooms()[:3]))
    data = np.asarray(image.dataobj)

    labels, counts = np.unique(data[data > 0], return_counts=True)
    return pd.Series(counts.astype(float) * voxel_volume,
                     index=[f'fs{int(label)}' for label in labels],
                     name=Path(segmentation).name)


def structural_features(fmriprep_dir: str | Path,
                        normalise: bool = True) -> pd.DataFrame:
    """Subjects x regions volume table from a fmriprep derivatives tree.

    Native-space segmentations only (no `space-` entity in the filename).
    With `normalise`, each subject's volumes are divided by their total — the
    cheap proxy for intracranial-volume correction, without which the leading
    'feature' of any structural model is head size.

    The label vocabulary is restricted to labels present in **every** subject
    with at least `min_voxels` voxels in the median subject. These fmriprep
    segmentations carry ~1,700 distinct ids per subject, over a hundred of them
    sub-10-voxel slivers; a feature that exists for some subjects at a handful
    of voxels is segmentation noise, not anatomy.
    """
    return _structural_features(fmriprep_dir, normalise)


def _structural_features(fmriprep_dir: str | Path,
                         normalise: bool = True,
                         min_voxel_volume_mm3: float = 100.0) -> pd.DataFrame:
    fmriprep_dir = Path(fmriprep_dir)
    paths = sorted(p for p in fmriprep_dir.glob(
        'sub-*/ses-*/anat/*_desc-aparcaseg_dseg.nii.gz')
        if 'space-' not in p.name)
    if not paths:
        raise FileNotFoundError(
            f'no native-space aparc+aseg segmentations under {fmriprep_dir}; '
            'fetch them with scripts/fetch_test_fixtures.py --what anat')

    rows = {}
    for path in paths:
        subject = path.parts[-4].removeprefix('sub-')
        rows[subject] = regional_volumes(path)

    table = pd.DataFrame(rows).T
    table.index.name = 'subject'

    # common vocabulary: in every subject, and not a sliver
    present_everywhere = table.notna().all(axis=0)
    big_enough = table.median(axis=0) >= min_voxel_volume_mm3
    table = table.loc[:, present_everywhere & big_enough]

    if normalise:
        table = table.div(table.sum(axis=1), axis=0)
    return table
