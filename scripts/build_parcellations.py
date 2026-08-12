"""Parcellate the resting runs many more ways than the paper's nine.

The multiverse currently varies the connectivity statistic six ways over one
6-network summary. That is six views of one 6x6 covariance matrix, and the
family-wise null showed it buys nothing. Spatial resolution is the axis far more
likely to carry signal a 6-network average destroys, so this script widens
*that* axis instead, using the preprocessed BOLD that is already on disk.

Two kinds of parcellation, and the distinction matters for how the results may
be read.

**Published atlases** are fixed maps defined by someone else on other data, so
they are independent of this sample by construction:

  schaefer{100,200,400,1000}-{7,17}   cortical, network labels included
  yeo{7,17}                           the network definitions Schaefer refines
  aal3, harvard-oxford                anatomical rather than functional
  difumo{256,512}, msdl                probabilistic (soft) parcellations

**Data-driven parcellations** are learned from BOLD, which makes them adaptive
but also makes provenance a live question:

  rena{100,200,500}   recursive nearest agglomeration -- spatially constrained,
                      and fast enough to refit per fold if one becomes a headline
  ward{200}           spatially constrained agglomerative clustering
  kmeans{200}         no spatial constraint, so parcels may be scattered
  canica{64}          group independent components (soft)
  dictlearn{64}       sparse dictionary learning (soft), usually cleaner than
                      ICA for connectivity work

The data-driven ones are fitted on all 32 participants **without labels**. That
is leakage-free in the sense the project's transfer test enforces -- permuting
the group labels leaves the parcellation bit-identical, because it never receives
them -- but it is still transductive: test-fold participants influenced the
parcel boundaries. Group-level ICA and clustering are conventionally fitted this
way, and for a *search* it is fine; anything promoted to a reported result must
be refitted inside training folds, which is not yet implemented here and is the
single caveat to carry forward from any data-driven row. ReNA is fast enough
(seconds per fit) that doing so is practical for whichever one survives.

Denoising matches the existing pipeline exactly (`load_confounds_strategy`,
'simple'), so the new timeseries are comparable with the nine already cached.

Output is one `timeseries_<name>.nc5` per parcellation in the same directory the
existing cache uses, so `explore_features.py` picks them up with no changes.
Already-written files are skipped, so the script is resumable.

    uv run python scripts/build_parcellations.py --tier 1
    uv run python scripts/build_parcellations.py --only schaefer200-7 rena200
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from nilearn.interfaces.fmriprep import load_confounds_strategy

warnings.filterwarnings('ignore')

FMRIPREP = Path('data/Julia2018/derivatives/fmriprep_2020')
CACHE = Path('data/Julia2018/derivatives/resting_timeseries')

# tier 1 is cheap label maskers, tier 2 data-driven hard parcellations,
# tier 3 the soft/probabilistic ones and group decompositions (slowest)
TIERS = {
    1: ('schaefer100-7', 'schaefer200-7', 'schaefer400-7', 'schaefer1000-7',
        'schaefer200-17', 'schaefer400-17', 'yeo7', 'yeo17', 'harvard-oxford'),
    2: ('rena100', 'rena200', 'rena500', 'ward200', 'kmeans200'),
    3: ('difumo256', 'difumo512', 'msdl', 'canica64', 'dictlearn64'),
}


def subject_files() -> tuple[list[str], list[str], list[str]]:
    images = sorted(FMRIPREP.glob(
        'sub-*/ses-rest/func/*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
    masks, subjects = [], []
    for image in images:
        mask = Path(str(image).replace('desc-preproc_bold', 'desc-brain_mask'))
        if not mask.exists():
            raise SystemExit(f'no brain mask beside {image.name}')
        masks.append(str(mask))
        subjects.append(image.name.split('_')[0].removeprefix('sub-'))
    return [str(i) for i in images], masks, subjects


def build_masker(name: str, images: list[str], mask: str):
    """(masker, region_names, network_names). Network names may be None."""
    from nilearn import datasets
    from nilearn.maskers import (
        NiftiLabelsMasker,
        NiftiMapsMasker,
    )

    common = {'mask_img': mask, 'standardize': 'zscore_sample',
              'memory': 'outputs/cache/nilearn', 'memory_level': 1}

    if name.startswith('schaefer'):
        body = name.removeprefix('schaefer')
        n_rois, networks = (int(p) for p in body.split('-'))
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois, yeo_networks=networks, resolution_mm=2)
        labels = [b.decode() if isinstance(b, bytes) else str(b)
                  for b in atlas.labels]
        # '7Networks_LH_Vis_1' -> network 'Vis'
        nets = [label.split('_')[2] if len(label.split('_')) > 2 else 'NA'
                for label in labels]
        return NiftiLabelsMasker(atlas.maps, **common), labels, nets

    if name.startswith('yeo'):
        # nilearn used to return a dict with 'thick_7'/'thick_17' keys; current
        # versions take the network count as an argument and return one map
        n = int(name.removeprefix('yeo'))
        atlas = datasets.fetch_atlas_yeo_2011(n_networks=n, thickness='thick')
        labels = [str(v) for v in atlas.labels[1:]]        # drop Background
        return NiftiLabelsMasker(atlas.maps, **common), labels, labels

    if name == 'harvard-oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        labels = [str(v) for v in atlas.labels[1:]]        # drop Background
        return NiftiLabelsMasker(atlas.maps, **common), labels, None

    if name == 'aal3':
        import pandas as pd
        table = pd.read_csv('data/atlases/AAL3v1.csv')
        labels = [str(v) for v in table.iloc[:, 1]]
        return (NiftiLabelsMasker('data/atlases/AAL3v1.nii.gz', **common),
                labels, None)

    if name.startswith('difumo'):
        dimension = int(name.removeprefix('difumo'))
        atlas = datasets.fetch_atlas_difumo(dimension=dimension, resolution_mm=2)
        labels = [str(v) for v in atlas.labels['difumo_names']]
        return NiftiMapsMasker(atlas.maps, **common), labels, None

    if name == 'msdl':
        atlas = datasets.fetch_atlas_msdl()
        return (NiftiMapsMasker(atlas.maps, **common),
                list(atlas.labels), list(atlas.networks))

    raise ValueError(f'unknown parcellation {name!r}')


def build_learned_masker(name: str, images: list[str], mask: str,
                         confounds: list, n_jobs: int):
    """Data-driven parcellations, fitted on the BOLD itself but never on labels."""
    from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker

    common = {'mask_img': mask, 'standardize': 'zscore_sample',
              'memory': 'outputs/cache/nilearn', 'memory_level': 1}

    if name.startswith(('rena', 'ward', 'kmeans')):
        from nilearn.regions import Parcellations

        method, k = ('rena', name.removeprefix('rena'))
        for prefix in ('ward', 'kmeans'):
            if name.startswith(prefix):
                method, k = prefix, name.removeprefix(prefix)
        parceller = Parcellations(
            method=method, n_parcels=int(k), mask=mask, standardize=False,
            memory='outputs/cache/nilearn', memory_level=1,
            n_jobs=n_jobs, random_state=0, verbose=0)
        parceller.fit(images, confounds=confounds)
        labels = [f'{name}_{i + 1}' for i in range(int(k))]
        return NiftiLabelsMasker(parceller.labels_img_, **common), labels, None

    if name.startswith(('canica', 'dictlearn')):
        from nilearn.decomposition import CanICA, DictLearning

        k = int(name.removeprefix('canica').removeprefix('dictlearn'))
        cls = CanICA if name.startswith('canica') else DictLearning
        model = cls(n_components=k, mask=mask, standardize=True,
                    memory='outputs/cache/nilearn', memory_level=1,
                    random_state=0, n_jobs=n_jobs, verbose=0)
        model.fit(images, confounds=confounds)
        labels = [f'{name}_{i + 1}' for i in range(k)]
        return NiftiMapsMasker(model.components_img_, **common), labels, None

    raise ValueError(f'unknown learned parcellation {name!r}')


def surviving_labels(masker, width: int, labels: list[str]) -> list[str]:
    """Which atlas regions this subject actually came back with.

    A NiftiLabelsMasker silently drops any region with no voxels under the
    subject's brain mask and packs the survivors to the left, so column k means
    a different region for different subjects. nilearn records the survivors in
    `region_names_`; maskers that never drop anything (NiftiMapsMasker) have no
    such attribute and are passed through.
    """
    surviving = getattr(masker, 'region_names_', None)
    if surviving is None:
        return list(labels[:width]) if len(labels) >= width else \
            [f'r{i}' for i in range(width)]
    if isinstance(surviving, dict):          # {column index: label}
        return [str(surviving[k]) for k in sorted(surviving)]
    return [str(v) for v in surviving]


def align_regions(rows: list[np.ndarray],
                  row_labels: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Restrict every subject to the regions present in all of them, in one order.

    This is the same defect, and the same remedy, as `Parcellation._align_regions`
    in the main pipeline: without it, column k is a different brain region for
    different participants and every connectivity estimate is meaningless.
    """
    common = set(row_labels[0])
    for names in row_labels[1:]:
        common &= set(names)
    ordered = [name for name in row_labels[0] if name in common]

    aligned = []
    for series, names in zip(rows, row_labels, strict=True):
        index = {name: i for i, name in enumerate(names)}
        aligned.append(series[:, [index[name] for name in ordered]])
    return np.stack(aligned), ordered


def parcellate(name: str, n_jobs: int, denoise: str = 'simple') -> Path:
    # the denoising strategy is part of the identity of the timeseries, so it
    # goes in the filename; 'simple' keeps the historical names unchanged
    suffix = '' if denoise == 'simple' else f'@{denoise}'
    destination = CACHE / f'timeseries_{name}{suffix}.nc5'
    if destination.exists():
        print(f'  {name}{suffix}: cached, skipping')
        return destination

    images, masks, subjects = subject_files()
    confounds, sample_masks = load_confounds_strategy(images, denoise)

    n_scans = confounds[0].shape[0]
    valid = set(np.hstack([s for s in sample_masks if s is not None]))
    keep = np.zeros(n_scans, dtype=bool)
    keep[np.array(sorted(valid)) - 1] = True

    started = time.perf_counter()
    learned = name.startswith(('rena', 'ward', 'kmeans', 'canica', 'dictlearn'))
    if learned:
        masker, labels, networks = build_learned_masker(
            name, images, masks[0], confounds, n_jobs)
    else:
        masker, labels, networks = build_masker(name, images, masks[0])

    rows, row_labels = [], []
    for image, mask, confound, sample_mask in zip(images, masks, confounds,
                                                  sample_masks, strict=True):
        masker.mask_img = mask
        series = masker.fit_transform(image, confounds=confound,
                                      sample_mask=sample_mask)
        if series.shape[0] > len(valid):
            series = series[keep]
        rows.append(series)
        row_labels.append(surviving_labels(masker, series.shape[1], labels))

    lengths = {r.shape[0] for r in rows}
    if len(lengths) != 1:
        raise RuntimeError(f'{name}: subjects returned different run lengths '
                           f'{lengths}')

    stacked, names = align_regions(rows, row_labels)
    if stacked.shape[2] < 2:
        raise RuntimeError(f'{name}: only {stacked.shape[2]} regions survive in '
                           f'every subject')

    data = {'timeseries': (('subject', 'timepoint', 'region'), stacked)}
    if networks is not None:
        # map through region *name*, not position: alignment above may have
        # dropped and reordered columns, so a positional slice would attach the
        # wrong network to every region after the first drop
        by_name = dict(zip(labels, networks, strict=False))
        assigned = [by_name.get(n, 'NA') for n in names]
        if sum(a != 'NA' for a in assigned) >= 2:
            data['network'] = (('region',), np.array(assigned))

    xr.Dataset(data, coords={'subject': subjects,
                             'region': names}).to_netcdf(destination)
    print(f'  {name}{suffix}: {stacked.shape} in '
          f'{time.perf_counter() - started:.0f}s -> {destination.name}')
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', type=int, nargs='*', default=[1])
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument(
        '--denoise', nargs='*', default=['simple'],
        help="nilearn confound strategies, e.g. simple scrubbing compcor. "
             "The motion analysis (manuscript 3.5) showed the groups differ in "
             "head motion and that 'simple' does not remove the "
             "motion-connectivity relationship, so this is the axis most likely "
             "to change the answer rather than merely widen the search.")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    Path('outputs/cache/nilearn').mkdir(parents=True, exist_ok=True)

    names = args.only or [n for t in args.tier for n in TIERS.get(t, ())]
    jobs = [(name, denoise) for denoise in args.denoise for name in names]
    print(f'building {len(jobs)} timeseries: {len(names)} parcellations x '
          f'{len(args.denoise)} denoising strategies '
          f'({", ".join(args.denoise)})\n')

    failed = []
    for name, denoise in jobs:
        try:
            parcellate(name, args.n_jobs, denoise)
        # SystemExit too: a per-parcellation abort must not kill the batch
        except (Exception, SystemExit) as error:
            failed.append((f'{name}@{denoise}', f'{type(error).__name__}: {error}'))
            print(f'  {name}@{denoise}: FAILED {type(error).__name__}: '
                  f'{str(error)[:160]}')

    print(f'\n{len(jobs) - len(failed)} of {len(jobs)} built')
    for name, error in failed:
        print(f'  failed: {name} -- {error[:200]}')


if __name__ == '__main__':
    main()
