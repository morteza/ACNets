"""gong2015, step 2: tissue segmentation and native-space regional GM volumes.

Runs under the *isolated* antspyx environment, not the project venv — antspyx
pins an older scipy and must not contaminate the analysis environment:

    /path/to/antsenv/bin/python scripts/gong2015_segment.py

Per subject: N4 bias correction -> brain extraction -> affine registration to
the MNI 2 mm template -> Atropos k-means 3-tissue segmentation in native space
-> atlas labels warped *into* native space with the inverse transform ->
GM volume per region = sum of GM probability x voxel volume inside each label.

Measuring in native space is the point of warping the atlas rather than the
brain: volumes are taken where the anatomy actually is, and no modulation
bookkeeping is needed. The registration is affine-only, which is the honest
budget choice for a pilot replication arm; the labelled limitation is that
region boundaries are only affinely accurate, and a SyN upgrade slots into the
same script if the arm graduates.

Writes data/gong2015/derivatives/gmv/sub-<id>_gmv.json (one per subject,
resumable) and gmv_{dosenbach2010,aal}.csv tables at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ants
import numpy as np

DERIVATIVES = Path('data/gong2015/derivatives')
NIFTI = DERIVATIVES / 'nifti'
ATLAS = DERIVATIVES / 'atlas'
GMV = DERIVATIVES / 'gmv'


def process_subject(t1_path: Path, template, atlases: dict) -> dict:
    subject = t1_path.name.removeprefix('sub-').removesuffix('_T1w.nii.gz')
    out_path = GMV / f'sub-{subject}_gmv.json'
    if out_path.exists():
        return json.loads(out_path.read_text())

    image = ants.image_read(str(t1_path))
    image = ants.n4_bias_field_correction(image)
    mask = ants.get_mask(image)

    registration = ants.registration(fixed=template, moving=image,
                                     type_of_transform='Affine')

    # k-means initialisation orders classes by intensity: on T1,
    # CSF < GM < WM, so GM is the middle class (index 2 of 'kmeans[3]')
    segmentation = ants.atropos(a=image, m='[0.1,1x1x1]', c='[5,0]',
                                i='kmeans[3]', x=mask)
    gm_probability = segmentation['probabilityimages'][1]

    voxel_volume = float(np.prod(image.spacing))
    record = {'subject': subject, 'voxel_volume_mm3': voxel_volume,
              'total_gm_mm3': float(gm_probability.numpy().sum() * voxel_volume)}

    for name, (label_image, label_ids) in atlases.items():
        native_labels = ants.apply_transforms(
            fixed=image, moving=label_image,
            transformlist=registration['invtransforms'],
            whichtoinvert=[True],
            interpolator='nearestNeighbor')
        label_data = native_labels.numpy()
        gm = gm_probability.numpy()
        volumes = {}
        for label in label_ids:
            volumes[str(label)] = float(gm[label_data == label].sum() * voxel_volume)
        record[name] = volumes

    GMV.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record))
    return record


def main() -> None:
    template = ants.image_read(str(ATLAS / 'mni_t1_2mm.nii.gz'))
    atlases = {}
    for name in ('dosenbach2010', 'aal'):
        label_image = ants.image_read(str(ATLAS / f'{name}_labels.nii.gz'))
        ids = np.loadtxt(ATLAS / f'{name}_labels.csv', delimiter=',',
                         skiprows=1, usecols=0, dtype=int).tolist()
        atlases[name] = (label_image, ids)

    t1s = sorted(NIFTI.glob('sub-*_T1w.nii.gz'))
    print(f'{len(t1s)} subjects', flush=True)

    records = []
    for index, t1 in enumerate(t1s, 1):
        records.append(process_subject(t1, template, atlases))
        print(f'  [{index}/{len(t1s)}] {t1.name}', flush=True)

    # assemble tables: rows are subjects, columns are region labels; volumes
    # normalised by each subject's total GM (head-size proxy)
    import csv
    for name in atlases:
        ids = [str(i) for i in atlases[name][1]]
        with open(DERIVATIVES / f'gmv_{name}.csv', 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['subject', 'total_gm_mm3', *ids])
            for record in records:
                total = record['total_gm_mm3']
                writer.writerow([record['subject'], total]
                                + [record[name][i] / total for i in ids])
    print('wrote gmv_dosenbach2010.csv and gmv_aal.csv', flush=True)


if __name__ == '__main__':
    sys.exit(main())
