"""Restore the DVC-tracked files the test suite needs, without pulling 11.5 GB.

Background. Before the `multimodal` merge, everything under `data/` was tracked
by a single root `data.dvc` covering 3355 files and 11.5 GB. That pointer was
replaced by per-dataset ones (`data/Julia2018.dvc`, `data/LEMON.dvc`), and the
files outside those two datasets stopped being tracked -- which is why roughly
half the test suite failed on missing paths.

The old pointer is still in git history, and its directory manifest is still on
the remote, so the files are recoverable. What makes this script worth having
rather than just running `dvc pull` is that the tests need about 830 MB of that
11.5 GB: the resting-state preproc images, their brain masks and confounds, and
the atlas definitions. Pulling the rest gets you every task run and every
fmriprep derivative for work that happens on the cluster anyway.

    uv run python scripts/fetch_test_fixtures.py --what atlases
    uv run python scripts/fetch_test_fixtures.py --what resting   # ~830 MB
    uv run python scripts/fetch_test_fixtures.py --what behaviour # ~4 MB
    uv run python scripts/fetch_test_fixtures.py --what dosenbach2007
    uv run python scripts/fetch_test_fixtures.py --what all --dry-run

Needs ssh access to the `aion-cluster` host in ~/.ssh/config.

`dosenbach2007` is a derivation rather than a fetch. `data/dosenbach2007/ROIs.csv`
is in no manifest, but the parcellated timeseries built *from* it is -- as
`julia2018/derivatives/resting_timeseries_v1/timeseries_dosenbach2007.nc`, whose
per-region variables carry the coordinates, network assignments and task
contrasts that the CSV held. `Parcellation.create_dataset` writes the atlas table
into every timeseries file it produces, so the atlas survives inside its own
output. Reconstructing the CSV from it recovers the original values rather than
inventing them.

Still not recoverable: `data/julia2018_raw/` and `data/julia2018_resting/`, both
pre-BIDS staging directories in no manifest. Their tests skip. Older manifests
were garbage-collected on the remote, so only the 2024-02-29 state survives.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Last commit before the merge that still carried the root pointer.
DATA_DVC_REV = 'c167cb6'
REMOTE_HOST = 'aion-cluster'
REMOTE_ROOT = '/work/projects/acnets/datasets/acnets/dvc/files/md5'
REPO = Path(__file__).resolve().parent.parent


def recover_manifest_hash() -> str:
    """The .dir object hash, read out of the deleted data.dvc in git history."""
    blob = subprocess.run(
        ['git', 'show', f'{DATA_DVC_REV}:data.dvc'],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in blob.splitlines():
        if 'md5:' in line and '.dir' in line:
            return line.split('md5:')[1].strip().removesuffix('.dir')
    raise SystemExit(f'no .dir hash in data.dvc at {DATA_DVC_REV}')


def fetch_manifest(dir_hash: str, cache: Path) -> list[dict]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        remote = f'{REMOTE_ROOT}/{dir_hash[:2]}/{dir_hash[2:]}.dir'
        subprocess.run(
            ['scp', '-o', 'BatchMode=yes', '-P', '8022', '-q',
             f'{REMOTE_HOST}:{remote}', str(cache)], check=True)
    return json.loads(cache.read_text())


DOSENBACH2007_SOURCE = (
    'julia2018/derivatives/resting_timeseries_v1/timeseries_dosenbach2007.nc')
DOSENBACH2007_COLUMNS = ['x', 'y', 'z', 'network', 'comp',
                         'sustained_Neg', 'sustained_Pos', 'cue', 'err']


def rebuild_dosenbach2007(manifest: list[dict], dry_run: bool) -> None:
    """Recover data/dosenbach2007/ROIs.csv out of a timeseries built from it."""
    import xarray as xr

    destination = REPO / 'data' / 'dosenbach2007' / 'ROIs.csv'
    if destination.exists():
        print(f'{destination.relative_to(REPO)} already present')
        return

    entry = next((e for e in manifest if e['relpath'] == DOSENBACH2007_SOURCE), None)
    if entry is None:
        raise SystemExit(f'{DOSENBACH2007_SOURCE} is not in the manifest')
    print(f'deriving from {DOSENBACH2007_SOURCE}')
    if dry_run:
        return

    staged = REPO / '.dvc' / 'tmp' / 'dosenbach2007.nc'
    staged.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{entry['md5'][:2]}/{entry['md5'][2:]}"
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-P', '8022', '-q',
                    f'{REMOTE_HOST}:{remote}', str(staged)], check=True)

    with xr.open_dataset(staged) as dataset:
        per_region = [v for v in dataset.data_vars if dataset[v].dims == ('region',)]
        table = dataset[per_region].to_dataframe()

    table.index.name = 'region'
    table = table[[c for c in DOSENBACH2007_COLUMNS if c in table.columns]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination)
    staged.unlink()

    print(f'wrote {destination.relative_to(REPO)}: {len(table)} regions')
    # Two region names read 'R frontal cortetrue' / 'L frontal cortetrue'. That
    # is in the stored data, not introduced here: the task-contrast columns mark
    # membership with 'x', and a replace('x', True) somewhere in the original
    # data preparation caught the name column too, turning 'cortex' into
    # 'cortetrue'. Reproduced verbatim, because these strings are the region
    # coordinate labels every existing artefact is keyed by; renaming them here
    # would silently fail to line up with anything already computed.
    mangled = [r for r in table.index if 'cortetrue' in str(r)]
    if mangled:
        print(f'  note: {len(mangled)} region names carry a pre-existing '
              f"'cortex' -> 'cortetrue' corruption: {mangled}")


def wanted(relpath: str, what: str) -> bool:
    """Which tracked paths each selection covers."""
    atlases = relpath.split('/')[0] in (
        'atlases', 'gordon2014', 'friedman2020', 'dosenbach2007', 'dosenbach2010')

    if what == 'atlases':
        return atlases

    if what == 'behaviour':
        low = relpath.lower()
        return any(k in low for k in ('behav', '/beh/', '_beh', 'events'))

    if what == 't1':
        # native-space preprocessed T1w, for running the gong2015 ANTs
        # segmentation on Julia2018 so both datasets land in one feature
        # space (see scripts/segment_t1.py). The MNI-resampled copy is
        # excluded: volumes must be measured where the anatomy actually is.
        return (relpath.startswith('julia2018/derivatives/fmriprep_2020/')
                and relpath.endswith('_desc-preproc_T1w.nii.gz')
                and 'space-' not in relpath)

    if what == 'anat':
        # fmriprep's anatomical segmentations: per-region volumes for the
        # structural modality of the multiview model (plan.md C4). ~32 files.
        return (relpath.startswith('julia2018/derivatives/fmriprep_2020/')
                and relpath.endswith('_desc-aparcaseg_dseg.nii.gz'))

    if what == 'resting':
        if not relpath.startswith('julia2018/'):
            return False
        if not relpath.startswith('julia2018/derivatives/fmriprep_2020/'):
            # BIDS root metadata, which pybids needs to read the layout at all
            return relpath.count('/') == 1
        if 'ses-rest' not in relpath:
            return False
        return (('space-MNI152NLin2009cAsym_desc-preproc_bold' in relpath)
                or ('space-MNI152NLin2009cAsym_desc-brain_mask' in relpath)
                or ('desc-confounds' in relpath))

    if what == 'all':
        return True

    raise SystemExit(f'unknown selection: {what}')


def fetch(entries: list[dict], dry_run: bool) -> None:
    """One ssh round trip: tar the cache objects, untar, rename to tracked paths.

    Per-file scp would be one connection per file, which for the ~700 small
    behavioural files costs far more in handshakes than in bytes.
    """
    todo = [e for e in entries if not (REPO / 'data' / e['relpath']).exists()]
    print(f'{len(entries)} tracked, {len(entries) - len(todo)} already present, '
          f'{len(todo)} to fetch')
    if dry_run or not todo:
        return

    stage = REPO / '.dvc' / 'tmp' / 'fixture-stage'
    if stage.exists():
        subprocess.run(['rm', '-rf', str(stage)], check=True)
    stage.mkdir(parents=True)

    objects = '\n'.join(
        f"{REMOTE_ROOT.lstrip('/')}/{e['md5'][:2]}/{e['md5'][2:]}" for e in todo)

    tar_remote = subprocess.Popen(
        ['ssh', '-o', 'BatchMode=yes', REMOTE_HOST, 'tar -cf - -C / -T -'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tar_local = subprocess.Popen(
        ['tar', '-xf', '-', '-C', str(stage)], stdin=tar_remote.stdout)
    tar_remote.stdin.write(objects.encode())
    tar_remote.stdin.close()
    tar_local.wait()

    restored = 0
    for e in todo:
        src = stage / REMOTE_ROOT.lstrip('/') / e['md5'][:2] / e['md5'][2:]
        if not src.exists():
            print(f'  missing on remote: {e["relpath"]}')
            continue
        dest = REPO / 'data' / e['relpath']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        restored += 1

    subprocess.run(['rm', '-rf', str(stage)], check=True)
    print(f'restored {restored} files')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--what', default='resting',
                        choices=['atlases', 'resting', 'behaviour', 'anat',
                                 't1', 'dosenbach2007', 'all'])
    parser.add_argument('--limit', type=int, default=None,
                        help='fetch at most this many files, for a cheap '
                             'feasibility check before committing to all of them')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    dir_hash = recover_manifest_hash()
    print(f'manifest {dir_hash}.dir, from data.dvc at {DATA_DVC_REV}')

    manifest = fetch_manifest(dir_hash, REPO / '.dvc' / 'tmp' / f'{dir_hash}.dir.json')
    print(f'manifest lists {len(manifest)} files')

    if args.what in ('dosenbach2007', 'all'):
        rebuild_dosenbach2007(manifest, args.dry_run)
        if args.what == 'dosenbach2007':
            return

    selected = [e for e in manifest if wanted(e['relpath'], args.what)]
    if not selected:
        raise SystemExit(f'nothing matched selection {args.what!r}')
    if args.limit:
        # keep both groups represented rather than taking the first n, which
        # would be all AVGP and would hide any group-dependent failure
        avgp = [e for e in selected if 'AVGP' in e['relpath']]
        nvgp = [e for e in selected if 'AVGP' not in e['relpath']]
        half = max(1, args.limit // 2)
        selected = avgp[:half] + nvgp[:args.limit - half]
        print(f'limited to {len(selected)} files '
              f'({sum("AVGP" in e["relpath"] for e in selected)} AVGP)')

    fetch(selected, args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
