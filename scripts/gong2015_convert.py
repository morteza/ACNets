"""gong2015, step 1: DICOM to NIfTI, plus the atlas label images.

Input: `data/gong2015/AVG-{expert,nonexpert}/sub_NN/` — one `Ax 3D FSPGR` T1
series per subject (verified from the DICOM headers; protocol
`Brain_gong_dota`, GE, 1 mm). Output:

    data/gong2015/derivatives/nifti/sub-<group><NN>_T1w.nii.gz
    data/gong2015/derivatives/participants.csv
    data/gong2015/derivatives/atlas/mni_t1_2mm.nii.gz
    data/gong2015/derivatives/atlas/{dosenbach2010,aal}_labels.nii.gz + .csv

The atlas images are built once, in MNI space: AAL from the tracked
`data/atlases/AAL3v1.nii.gz`, Dosenbach 2010 as 5 mm spheres stamped onto the
MNI 2 mm grid from the same coordinates the resting arm uses. Segmentation
(step 2) warps these into each subject's native space, so volumes are measured
where the anatomy is, not on the template.

Needs `dcm2niix` on PATH (brew install dcm2niix).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

RAW = Path('data/gong2015')
DERIVATIVES = RAW / 'derivatives'
NIFTI = DERIVATIVES / 'nifti'
ATLAS = DERIVATIVES / 'atlas'

GROUPS = {'AVG-expert': 'expert', 'AVG-nonexpert': 'nonexpert'}


def convert_all() -> pd.DataFrame:
    NIFTI.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, group in GROUPS.items():
        for subject_dir in sorted((RAW / folder).glob('sub_*')):
            number = subject_dir.name.removeprefix('sub_')
            subject = f'{group}{number}'
            target = NIFTI / f'sub-{subject}_T1w.nii.gz'
            if not target.exists():
                result = subprocess.run(
                    ['dcm2niix', '-z', 'y', '-b', 'y', '-f', f'sub-{subject}_T1w',
                     '-o', str(NIFTI), str(subject_dir)],
                    capture_output=True, text=True)
                if not target.exists():
                    print(f'  FAILED {subject}: {result.stdout[-200:]}')
                    continue
            shape = nib.load(target).shape
            rows.append({'subject': subject, 'group': group,
                         'source': str(subject_dir), 'shape': str(shape)})
    table = pd.DataFrame(rows)
    table.to_csv(DERIVATIVES / 'participants.csv', index=False)
    return table


def build_atlas_images() -> None:
    """MNI template + label images the segmentation step will warp to native."""
    from nilearn.datasets import fetch_coords_dosenbach_2010, load_mni152_template

    ATLAS.mkdir(parents=True, exist_ok=True)

    template = load_mni152_template(resolution=2)
    template_path = ATLAS / 'mni_t1_2mm.nii.gz'
    template.to_filename(template_path)

    # --- Dosenbach 2010: 5 mm spheres on the 2 mm grid --------------------
    atlas = fetch_coords_dosenbach_2010(ordered_regions=False)
    coords = atlas['rois'][['x', 'y', 'z']].to_numpy()
    networks = [str(n) for n in atlas['networks']]

    shape, affine = template.shape, template.affine
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'),
                    axis=-1)
    world = nib.affines.apply_affine(affine, grid)

    labels = np.zeros(shape, dtype=np.int16)
    for index, centre in enumerate(coords, start=1):
        distance = np.linalg.norm(world - centre, axis=-1)
        labels[(distance <= 5.0) & (labels == 0)] = index
    nib.Nifti1Image(labels, affine).to_filename(
        ATLAS / 'dosenbach2010_labels.nii.gz')
    pd.DataFrame({
        'label': np.arange(1, len(coords) + 1),
        'region': [str(r) for r in atlas['rois'].index],
        'network': networks,
    }).to_csv(ATLAS / 'dosenbach2010_labels.csv', index=False)

    # --- AAL: resample the tracked label image to the template grid -------
    from nilearn.image import resample_to_img

    aal = resample_to_img(str(Path('data/atlases/AAL3v1.nii.gz')), template,
                          interpolation='nearest',
                          force_resample=True, copy_header=True)
    aal.to_filename(ATLAS / 'aal_labels.nii.gz')
    aal_table = pd.read_csv('data/atlases/AAL3v1.csv').dropna(subset=['index'])
    pd.DataFrame({
        'label': aal_table['index'].astype(int),
        'region': aal_table['region'],
    }).to_csv(ATLAS / 'aal_labels.csv', index=False)

    print(f'atlas images: dosenbach2010 ({len(coords)} spheres), '
          f'aal ({len(aal_table)} labels), template {template.shape}')


def main() -> None:
    table = convert_all()
    counts = table['group'].value_counts().to_dict()
    print(f'converted {len(table)} subjects: {counts}')
    build_atlas_images()
    (DERIVATIVES / 'provenance.json').write_text(json.dumps({
        'note': ('Group labels come from the AVG-expert / AVG-nonexpert '
                 'directory names. Whether this is the structural arm of '
                 'Gong et al. (2015, Sci Rep 5:9763) is plausible (protocol '
                 'Brain_gong_dota, matching design) but UNCONFIRMED -- '
                 'plan.md D1 assigns that confirmation to the authors.'),
        'n_by_group': counts,
    }, indent=2))


if __name__ == '__main__':
    main()
