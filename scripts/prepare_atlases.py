"""Fetch volumetric atlases with nilearn and stage them for the segmentation step.

No FreeSurfer anywhere in this pipeline, and none needed: `nilearn` fetches the
atlases, `antspyx` does bias correction, registration and tissue segmentation,
and both are pip-installable. The only reason there are two environments is that
antspyx pins an older scipy, not that anything external is required.

**Use a volumetric parcellation, not a sphere atlas.** This is the whole reason
the script exists. The gong2015 and Julia2018 structural volumes would not agree
at all under Dosenbach 2010 -- their regional profiles correlated at **-0.41**
across three different masking and registration pipelines -- and the cause was
neither masking nor registration but the atlas itself. Dosenbach 2010 is 160
small spheres placed at coordinates; with affine registration between different
acquisitions, a 10 mm sphere can miss its target, land in white matter or sit
half outside the brain, so the grey-matter volume it picks up is close to
arbitrary. Swapping to AAL, whose parcels are large and contiguous, takes the
same two datasets to **r = +0.87** with no other change.

Sphere atlases are fine for functional connectivity, where the timeseries is
averaged over whatever the sphere contains and small displacements mostly add
noise. They are unsuitable for regional volumetry across datasets.

Atlases staged here, all volumetric:

    aal                 116 anatomical parcels, the current default
    harvard-oxford      cortical, probabilistic thresholded to maxprob
    schaefer{100,200,400}   functional parcels with Yeo network labels
    yeo{7,17}           the coarse network parcellation itself
    juelich             cytoarchitectonic, a different labelling principle

    uv run python scripts/prepare_atlases.py --atlases aal schaefer200 yeo7
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from nilearn import datasets, image

warnings.filterwarnings('ignore')

OUT = Path('data/atlases/prepared')


def _labels_from(maps, names: list[str]) -> tuple:
    """(image, [(label_id, region, network)]) for a label image."""
    data = np.asarray(image.load_img(maps).dataobj)
    present = sorted(int(v) for v in np.unique(data) if v != 0)
    rows = []
    for label in present:
        index = label - 1
        name = names[index] if 0 <= index < len(names) else f'region{label}'
        # Schaefer names look like '7Networks_LH_Vis_1'; everything else has no
        # network and gets 'NA' rather than a guess
        parts = name.split('_')
        network = parts[2] if len(parts) > 3 and parts[0].endswith('Networks') else 'NA'
        rows.append((label, name, network))
    return image.load_img(maps), rows


def fetch(name: str):
    if name == 'aal':
        atlas = datasets.fetch_atlas_aal()
        data = np.asarray(image.load_img(atlas.maps).dataobj)
        present = sorted(int(v) for v in np.unique(data) if v != 0)
        lookup = {int(i): n for i, n in zip(atlas.indices, atlas.labels,
                                            strict=False)}
        rows = [(label, lookup.get(label, f'region{label}'), 'NA')
                for label in present]
        return image.load_img(atlas.maps), rows

    if name == 'harvard-oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        return _labels_from(atlas.maps, [str(v) for v in atlas.labels[1:]])

    if name.startswith('schaefer'):
        n_rois = int(name.removeprefix('schaefer'))
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois, yeo_networks=7, resolution_mm=2)
        names = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in atlas.labels]
        # nilearn returns 201 names for 200 parcels, with 'Background' first,
        # while the image is labelled 1..200. `_labels_from` indexes with
        # label - 1, so leaving Background in shifts every parcel's name and
        # network by one -- silently, since the volumes are keyed by label id
        # and stay correct while only the *interpretation* goes wrong.
        if names and 'background' in names[0].lower():
            names = names[1:]
        return _labels_from(atlas.maps, names)

    if name.startswith('yeo'):
        n = int(name.removeprefix('yeo'))
        atlas = datasets.fetch_atlas_yeo_2011(n_networks=n, thickness='thick')
        return _labels_from(atlas.maps, [str(v) for v in atlas.labels[1:]])

    if name == 'juelich':
        atlas = datasets.fetch_atlas_juelich('maxprob-thr25-2mm')
        return _labels_from(atlas.maps, [str(v) for v in atlas.labels[1:]])

    raise SystemExit(f'unknown atlas {name!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlases', nargs='*',
                        default=['aal', 'harvard-oxford', 'schaefer100',
                                 'schaefer200', 'yeo7', 'yeo17'])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.atlases:
        try:
            labels_img, rows = fetch(name)
        except Exception as error:
            print(f'  {name}: FAILED {type(error).__name__}: {str(error)[:120]}')
            continue

        labels_img.to_filename(OUT / f'{name}_labels.nii.gz')
        with open(OUT / f'{name}_labels.csv', 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['label', 'region', 'network'])
            writer.writerows(rows)

        data = np.asarray(labels_img.dataobj)
        sizes = [int((data == label).sum()) for label, _, _ in rows]
        zooms = labels_img.header.get_zooms()[:3]
        volumes = np.array(sizes) * float(np.prod(zooms)) / 1000
        print(f'  {name:16s} {len(rows):4d} parcels   '
              f'median {np.median(volumes):6.1f} cm3   '
              f'min {volumes.min():5.2f}   '
              f'{"SPHERE-LIKE, unsuitable" if np.median(volumes) < 2 else "volumetric"}')

    print(f'\nstaged in {OUT}/ -- pass to segment_t1.py with --atlas-dir')


if __name__ == '__main__':
    main()
