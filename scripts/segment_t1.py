"""Native-space regional grey-matter volumes, for any dataset, one pipeline.

`gong2015_segment.py` did this for gong2015 with its paths hard-coded. Putting
Julia2018 through the *same* pipeline is what makes a cross-dataset analysis
possible at all: the two datasets currently sit in incompatible feature spaces --
gong2015 in atlas-based GMV from ANTs, Julia2018 in FreeSurfer `aparc+aseg`
labels from fmriprep -- and volumes from two different segmentation tools are
not comparable columns.

Why this matters more than pooling. Two datasets in one feature space allow
**external validation**: train on gong2015's 47 participants, predict
Julia2018's 32, or the reverse. A single prediction on a genuinely independent
sample needs no multiverse correction, which is the one thing that converts a
Tier 3 claim into a Tier 1 one without pre-registration
(`docs/paper/publication_strategy.md`).

Runs under an isolated antspyx environment, not the project venv -- antspyx
pins an older scipy and must not contaminate the analysis environment:

    /path/to/antsenv/bin/python scripts/segment_t1.py \\
        --nifti data/Julia2018/derivatives/fmriprep_2020 \\
        --pattern '*_desc-preproc_T1w.nii.gz' \\
        --out data/Julia2018/derivatives/gmv --name julia2018

Per subject: N4 bias correction, brain extraction, affine registration to the
MNI 2 mm template, Atropos k-means 3-tissue segmentation in native space, then
the atlas labels warped *into* native space with the inverse transform. Grey
matter volume per region is the sum of GM probability inside each label times
the voxel volume. Measuring in native space is the point of warping the atlas
rather than the brain: volumes are taken where the anatomy actually is.

Resumable -- a subject whose JSON already exists is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import ants
import numpy as np

ATLAS = Path('data/gong2015/derivatives/atlas')


def subject_id(path: Path) -> str:
    """`sub-AVGP01_ses-rest_desc-preproc_T1w.nii.gz` -> `AVGP01`."""
    name = path.name
    for suffix in ('_desc-preproc_T1w.nii.gz', '_T1w.nii.gz'):
        name = name.removesuffix(suffix)
    name = name.removeprefix('sub-')
    return name.split('_')[0]


def brain_mask(image, registration, template_mask):
    """Skull-strip by warping an MNI brain mask into the subject's space.

    `ants.get_mask` is an intensity heuristic, not a brain extractor, and it
    does not remove skull and scalp from a whole-head T1. On gong2015 it
    returned 2,127 cm3 against a plausible brain of 1,200-1,600, so the
    three-class Atropos step was separating background / skull-and-scalp /
    brain rather than CSF / grey / white -- and the "grey matter" volumes it
    produced were contaminated. That defect is recorded in
    `docs/paper/decodability_ceiling.md` section 16.

    Warping a template brain mask through the registration that is computed
    anyway is robust, needs no deep-learning dependency, and works whether or
    not the input has already been stripped: intersecting with the intensity
    mask keeps it tight for inputs like fmriprep's, which are stripped already.

    Used for a first, coarse registration only; `process` then re-registers
    brain-to-brain with this mask applied, because the registration target
    itself has to be the same kind of object in both datasets.

    Two details decide whether this works, and the first attempt got both
    wrong. nilearn's default MNI brain mask is *liberal* -- 1,883 cm3 at
    threshold 0.2, about 30% larger than a brain -- and an affine warp onto a
    larger head inflates it further, so masks came back at 2,200 cm3, no better
    than the `get_mask` they replaced. The mask is therefore built at threshold
    0.6 (1,493 cm3, conservative), and the warped result is checked against a
    plausible range rather than trusted.
    """
    warped = ants.apply_transforms(
        fixed=image, moving=template_mask,
        transformlist=registration['invtransforms'],
        whichtoinvert=[True], interpolator='nearestNeighbor')
    # intersect with a light intensity mask so background inside the warped
    # template mask does not enter the segmentation
    intensity = ants.get_mask(image, cleanup=2)
    combined = warped * intensity
    if float(combined.numpy().sum()) < 0.2 * float(warped.numpy().sum()):
        # registration probably failed; fall back to the template mask alone
        # rather than silently segmenting a sliver of brain
        return warped
    return combined


def process(t1_path: Path, template, template_mask, atlases: dict,
            out_dir: Path) -> dict:
    subject = subject_id(t1_path)
    destination = out_dir / f'sub-{subject}_gmv.json'
    if destination.exists():
        return json.loads(destination.read_text())

    image = ants.image_read(str(t1_path))
    image = ants.n4_bias_field_correction(image)

    # Register brain-to-brain, not brain-to-head. gong2015's images are whole
    # head; Julia2018's fmriprep output is already skull-stripped. Affinely
    # registering a stripped brain to a whole-head template scales the brain up
    # to fill the head, so the atlas lands in systematically the wrong place --
    # which is why the two datasets' regional profiles correlated at -0.39 even
    # after the skull-stripping defect was fixed. Masking both sides first makes
    # the registration target the same object in both datasets.
    coarse = ants.registration(fixed=template, moving=image,
                               type_of_transform='Affine')
    mask = brain_mask(image, coarse, template_mask)

    registration = ants.registration(fixed=template * template_mask,
                                     moving=image * mask,
                                     type_of_transform='Affine')

    # k-means initialisation orders classes by intensity: on T1,
    # CSF < GM < WM, so GM is the middle class
    segmentation = ants.atropos(a=image, m='[0.1,1x1x1]', c='[5,0]',
                                i='kmeans[3]', x=mask)
    gm_probability = segmentation['probabilityimages'][1]

    voxel_volume = float(np.prod(image.spacing))
    mask_volume = float(mask.numpy().sum()) * voxel_volume / 1000
    if not 900 < mask_volume < 2000:
        print(f'    WARNING {subject}: brain mask {mask_volume:.0f} cm3 is '
              f'outside the plausible 900-2000 range', flush=True)
    record = {'subject': subject, 'voxel_volume_mm3': voxel_volume,
              'brain_mask_cm3': mask_volume,
              'total_gm_mm3': float(gm_probability.numpy().sum() * voxel_volume)}

    for name, (label_image, label_ids) in atlases.items():
        native_labels = ants.apply_transforms(
            fixed=image, moving=label_image,
            transformlist=registration['invtransforms'],
            whichtoinvert=[True], interpolator='nearestNeighbor')
        labels = native_labels.numpy()
        gm = gm_probability.numpy()
        record[name] = {str(label): float(gm[labels == label].sum() * voxel_volume)
                        for label in label_ids}

    out_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nifti', required=True,
                        help='directory searched recursively for T1 images')
    parser.add_argument('--pattern', default='*_T1w.nii.gz')
    parser.add_argument('--out', required=True)
    parser.add_argument('--name', required=True,
                        help='dataset name, used in the output table filenames')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--atlas-dir', default='data/atlases/prepared',
                        help='directory of staged atlases from '
                             'scripts/prepare_atlases.py')
    parser.add_argument('--atlases', nargs='*',
                        default=['aal', 'harvard-oxford', 'schaefer200'],
                        help='volumetric parcellations only. A sphere atlas '
                             'such as dosenbach2010 is unsuitable here: its '
                             'regional volumes did not agree between datasets '
                             'at all (r = -0.41) where AAL reaches +0.87')
    args = parser.parse_args()

    template = ants.image_read(str(ATLAS / 'mni_t1_2mm.nii.gz'))
    template_mask = ants.image_read(str(ATLAS / 'mni_brain_mask.nii.gz'))

    atlas_dir = Path(args.atlas_dir)
    atlases = {}
    for name in args.atlases:
        labels_path = atlas_dir / f'{name}_labels.nii.gz'
        if not labels_path.exists():
            raise SystemExit(f'{labels_path} missing; run '
                             f'scripts/prepare_atlases.py --atlases {name}')
        atlases[name] = (
            ants.image_read(str(labels_path)),
            np.loadtxt(atlas_dir / f'{name}_labels.csv', delimiter=',',
                       skiprows=1, usecols=0, dtype=int).tolist())
    print(f'atlases: {", ".join(atlases)}', flush=True)

    t1s = sorted(Path(args.nifti).rglob(args.pattern))
    # the MNI-resampled copy measures the template's anatomy as much as the
    # participant's, so it is never the right input here
    t1s = [p for p in t1s if 'space-' not in p.name]
    if args.limit:
        t1s = t1s[:args.limit]
    if not t1s:
        raise SystemExit(f'no T1 images matching {args.pattern!r} under {args.nifti}')
    print(f'{len(t1s)} subjects', flush=True)

    out_dir = Path(args.out)
    records = []
    for index, t1 in enumerate(t1s, 1):
        records.append(process(t1, template, template_mask, atlases, out_dir))
        print(f'  [{index}/{len(t1s)}] {subject_id(t1)}', flush=True)

    for name in atlases:
        ids = [str(i) for i in atlases[name][1]]
        table = out_dir.parent / f'gmv_{args.name}_{name}.csv'
        with open(table, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['subject', 'total_gm_mm3', *ids])
            for record in records:
                total = record['total_gm_mm3']
                writer.writerow([record['subject'], total]
                                + [record[name][i] / total for i in ids])
        print(f'wrote {table}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
