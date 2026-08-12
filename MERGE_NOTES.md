# Merge notes: `multimodal` into `main`

The merge landed as `6dd7fb7`, a two-parent merge commit. The duplication and the
environment breakage it left behind have since been cleaned up; what remains outstanding
is the DVC reconciliation, which needs the data. Delete this file once that is done.

## Still outstanding

### Reconcile DVC before any `dvc pull`

Root `data.dvc` was removed in favour of per-dataset pointers, `data/Julia2018.dvc` and
`data/LEMON.dvc`, which are now authoritative. The old `data.dvc` tracked 11.5 GB across
3355 files as a single directory, which overlaps the per-dataset outputs and would make DVC
refuse to operate. Re-running `dvc add` needs the data present and HPC access, so it was
left for you.

On a machine with the data:

```bash
dvc status                 # expect complaints about missing or changed outs
dvc add data/Julia2018     # only if hashes no longer match
dvc add data/LEMON
dvc push -r aion-cluster
```

Anything in the old coarse `data/` tree outside `Julia2018/` and `LEMON/` is no longer
tracked by DVC. All of it that the tests need has since been recovered — see "Recovering the
DVC-tracked test fixtures" below — taking the suite from **26 failed / 202 passed** to
**0 failed / 225 passed / 3 skipped**.

The three skips need inputs that were never tracked in any manifest, and both are tests of
superseded code rather than of the current pipeline:

| skipped test | wants | status |
|---|---|---|
| `test_load_connectivity` | `data/julia2018_resting/connectivity_<atlas>.nc` | input format of `datasets/__julia2018_old.py`, read only by that module and two archived notebooks |
| `test_rest2bids`, `test_task2bids` | `data/julia2018_raw/` plus dcm2niix | pre-BIDS DICOM staging tree; dcm2niix is a system binary, not a dependency |

Both skip conditionally on the directory being absent, so they run on a machine that has it.

Note the case distinction, which matters on the cluster even though macOS hides it:
`data/Julia2018/` holds the derived timeseries; `data/julia2018/` is the BIDS tree, of which
a deliberate subset is now present.

`models.dvc` and `outputs.dvc` at the root were untouched.

### Delete the remote branch once satisfied

```bash
git push origin --delete multimodal
```

## What the merge did

Two root commits with no common ancestor, so this required
`--allow-unrelated-histories`:

| | root | last commit | commits |
|---|---|---|---|
| `main` | `18d916d` 2020-09-02 | `c167cb6` 2024-03-06 | 879 |
| `multimodal` | `cb1e629` 2024-03-25 | `3b83933` 2024-09-04 | 38 |

44 changes: 36 additions, 7 modifications, 1 deletion. Only 7 files collided. Both trees
are kept, `src/acnets/` for the classic pipeline and `src/multimodal/` for the deep
learning line.

Two of the seven resolutions were load-bearing rather than cosmetic. The multimodal
branch set `python_files = "*_test.py"`, which would have silently hidden the entire
`tests/` suite inherited from main; `pyproject.toml` now accepts both patterns. And main's
`.gitignore` ignored `/data` wholesale, which would have made the new per-dataset DVC
pointers invisible to git; it is now an allowlist, with a comment explaining why not to
simplify it back.

Dependency management has since moved to PEP 621 `[project]` in `pyproject.toml`, and
`environment.yml` has been deleted. `pyspi` was evaluated and dropped; the reasoning is
recorded in a comment in `pyproject.toml`.

## Resolved

### The environment could not be solved at all, and has since moved to uv

`pixi install` failed outright, so nothing in the repo could be run. The project now uses
**uv on Python 3.14**, pinned in `.python-version`; pixi, conda, mamba and `environment.yml`
are all gone and `pyproject.toml` is the only place dependencies are declared. 3.15 resolves
everything except torch, which has no 3.15 ABI wheels yet.

The one constraint that survives the move, because it is a genuine PyPI-level conflict rather
than a packaging artefact:

- `shap` needs `numba`, and numba's numpy ceiling trails numpy by a release or two. Left
  alone the resolver satisfies that ceiling by walking numba back to 0.53, which drags in
  `llvmlite` 0.36 — an sdist that will not build on any modern Python. `numba>=0.62` floors
  it, which is why numpy settles on 2.4 rather than the newest.

Three things that were pixi-specific and are now moot: the conda-forge pins on the compiled
stack (`hdf5`, `h5py`, `netcdf4`, …), added because pixi's PyPI resolver kept choosing an
h5py sdist it could not compile; the `numpy < 2.5` conda cap; and a `dev` pixi feature
mirroring `[dependency-groups].dev`, which pixi did not read. One did have to be carried
over explicitly: **`h5py` is now a declared dependency**, because `h5netcdf` treats its
backends as optional and everything in `data/` is opened with `engine='h5netcdf'`.

`pyinform` moved to a `te` extra and stayed there. On 3.14 it installs, having previously
refused to build at all, but the `libinform` it vendors is x86_64 only, so on this machine it
fails at call time instead. **`transfer_entropy` specifications do not run on an arm64 mac
either way** — run them on the cluster, or precompute the matrices and read them back through
`Specification`'s `ext:` prefix. The agreement analysis therefore covers six of the seven
statistics.

`.pixi/` is still on disk and is now orphaned; `rm -rf .pixi` when convenient.

### Duplicated modules

The three preprocessing transformers were near-forks and are now a single implementation.
`src/multimodal/preprocessing/` re-exports from `acnets.pipeline`; both import paths return
the same objects. The forks had drifted in both directions and each side kept what it had
right:

- the `multimodal` copies recorded `timeseries_aggregation_strategy` and
  `connectivity_kind` on the returned dataset, which the `acnets` copies did not. Ported
  across — `notebooks/playground/8 Nilearn Features.ipynb` reads the latter.
- the `multimodal` `ConnectivityExtractor` had been narrowed to nilearn's
  `ConnectivityMeasure`, silently dropping `chatterjee` and `transfer_entropy`. Those two
  are half the point of the multiverse grid, so the `acnets` version wins.

The four empty stubs are deleted, except `src/multimodal/__init__.py`, which is a
legitimate package marker.

The remaining collisions went the same way once the deep-learning code moved to plain PyTorch
(below), since two of the three only existed because the two copies had been written against
different frameworks:

| module | resolution |
|---|---|
| `gan.py` | one implementation in `acnets/deep/`; `multimodal/models/` re-exports it. The semi-supervised design already covered the unconditional case, so there was nothing to merge. |
| `lemon.py`, `julia2018*.py` | one class per dataset in `acnets/datasets/`; `multimodal/data/` re-exports. The `acnets` copies parcellated the raw BIDS tree, the `multimodal` copies read the cached netCDF; the survivors do both, cache first. |
| `utils.py` | genuinely unrelated — a GridSearchCV progress bar against a training progress bar. Left alone. |

A fourth copy turned up during the rewrite: the `aal` branch of both data modules inlined
`parcellations/aal.load_masker`, and had lost `resampling_target='labels'` on the way, so that
atlas resampled to the data grid there and to the label grid everywhere else. Both now call
the shared function.

`src/acnets/deep/` holds all the models. `src/multimodal/models/` keeps only
`hierarchical_temporal.py`, the learnable-parcellation stub described in Tier 2 of the roadmap.

### Stale references

- `datasets/lemon.py` pointed at `data/mpi-lemon/{atlas}_timeseries.nc`, the pre-merge
  layout. Now `data/LEMON/timeseries_{atlas}.nc`, which is what the DVC pointer tracks.
- `datasets/lemon.py` imported `from src.acnets.parcellations import ...`. A `src.`-prefixed
  import inside the package resolves to a *second copy* of the module under a different
  name, and breaks entirely once the package is installed as a wheel. The inherited
  `tests/` had eleven more; all now import `acnets.` directly, which is what
  `pythonpath = ["src"]` is for.
- `preprocessing/utils.py` ran `raise "..."` at import time when nipype was missing — not a
  valid exception, and it made the whole `acnets.preprocessing` package unimportable,
  taking `preprocessing/beh.py` down with it. Now deferred to the one function that needs
  it.
- `acnets/datasets/__init__.py` eagerly imported both data modules, so `acnets.datasets`
  required torch. Now lazy via `__getattr__`.
- Three parcellation loaders passed `legacy_format=` to nilearn fetchers, which nilearn
  0.13 removed. The non-legacy shape they selected is now the only one returned, so the
  argument is simply dropped.
- `specification.py` and `registry.py` both cite a note in `pyproject.toml` on the pyspi
  evaluation that the merge lost. Reconstructed from those two call sites — **check that it
  says what you meant** before citing it in the manuscript.

### `ExtraConnectivityMeasure.transform` refitted on its input

`transform` called `super().fit_transform(...)`. Harmless for the per-subject statistics,
wrong for `tangent`, which projects onto a group mean covariance learned in `fit`: each
test fold was getting its own reference point. Now calls `super().transform(...)`.
`src/acnets/multiverse/runner_test.py::test_tangent_reference_is_not_refitted_on_the_test_fold`
guards it.

### Bootstrap interval in `agreement.py` excluded its own point estimate

`agreement_matrix` resamples specifications with replacement, then masked only the literal
diagonal of the resampled matrix. At n = 12 essentially every draw repeats a specification,
and each repeat puts a self-comparison — a perfect 1.0 — at an off-diagonal position. The
interval was biased up far enough to sit entirely above the mean it brackets: the
random-network control reported 0.498 with a CI of [0.501, 0.604]. The existing test missed
it because its fixture agreed so strongly there was no headroom for the bias to show.

### Keras and PyTorch Lightning replaced with plain PyTorch

Seventeen files across `acnets/deep/`, `acnets/datasets/`, `multimodal/models/` and
`multimodal/data/`. What the two frameworks were actually being used for was an epoch loop,
device placement, TensorBoard scalars, and one checkpoint at the end of pretraining; that is
now `acnets/deep/training.py`, at roughly the size the Lightning subclasses took to configure.

Models plug in by implementing `compute_loss(batch, device)`, or `training_step(batch,
optimizers, device)` when they drive two optimizers themselves. `FineTunable` keeps the
pretrain/finetune protocol. `torchmetrics.Accuracy` became a function, which cannot forget to
be `.reset()` the way the old call sites sometimes did.

Four behaviour changes, all deliberate:

- **Device defaults to CPU**, opt in with `ACNETS_DEVICE=mps` or `resolve_device('auto')`.
  This replaces the hack in `3b83933` that reached into Keras global state to undo Keras's
  own MPS selection. MPS silently computes some ops in float32 and has disagreed with CPU on
  LSTM and ConvTranspose1d, both used here. `tests/keras3_disable_mps_test.py` is deleted;
  the concern it documented is now `TestResolveDevice`.
- **`GAN`'s discriminator returns logits.** It ended in a `Sigmoid` whose output went to
  `F.cross_entropy`, which applies its own `log_softmax` — the range was being squashed
  before a second normalisation, flattening the gradients.
- **`ConditionalGAN`'s generator target is `y + 1`.** Class 0 means "generated", and the
  Keras version one-hot encoded `y_real` over `n_classes + 1` without the shift, so the
  generator was rewarded for producing samples the discriminator labelled as the class below
  the requested one. For class 1 that target was "look fake".
- **`MultiHeadModel.dropout_accuracy` restores the previous train/eval mode.** It called
  `enable_dropout()` and never turned it off, so everything evaluated after the first
  validation pass ran with dropout live.

`num_workers` now defaults to 0 rather than `os.cpu_count() - 1`. The datasets fit in memory;
the old default spawned eleven processes to iterate 32 subjects.

The deep-learning code had no tests at all before this. It now has 37, in
`deep/training_test.py`, `deep/fine_tunable_test.py` and `multimodal/utils_test.py`.

### `_r_squared` in `variance.py` divided by a rounding error

Guarding constant `y` with `total == 0.0` is not enough: the mean of n identical floats is not
always exactly that float, and the error grows with n. At the 210 specifications of the
balanced grid the mean of 210 copies of 0.65 came out one ulp high, the total sum of squares
landed at 2.6e-30 instead of 0, and dividing a comparably tiny residual by it reported a
partial variance of 1181 where the answer is 0. The guard is now scale-aware. It surfaced on
the numpy bump and would have surfaced eventually anyway; the failure is silent and reads as
an enormous effect.

## Recovering the DVC-tracked test fixtures

The files the tests were missing turned out to be recoverable. The deleted root `data.dvc`
still exists in git history at `c167cb6`, its directory manifest
(`4c2f4a53f0eb55ba1b5026251166aed8.dir`, 3355 files, 11.5 GB) is still on `aion-cluster`, and
every object it names is still there. `scripts/fetch_test_fixtures.py` automates the whole
route: recover the hash from git, fetch the manifest, select, restore.

Restored, and now tracked by their own per-dataset pointers:

| what | files | size |
|---|---|---|
| `data/atlases/`, `gordon2014/`, `friedman2020/`, `dosenbach2007/`, `dosenbach2010/` | 14 | 34 MB |
| `data/julia2018/` resting subset: preproc bolds, brain masks, confounds, BIDS root metadata | 262 | 830 MB |
| `data/julia2018/` behavioural: `sourcedata/raw_behavioral/` plus the per-subject `beh/` and `events` files | 687 | 4 MB |

The resting subset is the 830 MB of the 11.36 GB `julia2018/` tree that the tests actually
read — what `Parcellation._get_fmriprep_files` globs, plus the confounds. `data/julia2018/`
is deliberately **not** under `dvc add`: what is on disk is a deliberate subset, and a pointer
for it would claim otherwise. Use the script, or `dvc pull` the real thing on the cluster.

### `data/dosenbach2007/ROIs.csv`, recovered from its own output

This one is in no manifest, and the first pass wrote it off. It is recoverable anyway,
because `Parcellation.create_dataset` merges the atlas table into every timeseries file it
writes — so the atlas survives inside its own product. The manifest holds
`julia2018/derivatives/resting_timeseries_v1/timeseries_dosenbach2007.nc`, whose per-region
variables are exactly the CSV's columns: `x`, `y`, `z`, `network`, `comp` and the four task
contrasts, across all 39 ROIs. `scripts/fetch_test_fixtures.py --what dosenbach2007`
reconstructs it. These are the stored values, not values invented to fit.

**One thing to know about that file.** Two regions are named `R frontal cortetrue` and
`L frontal cortetrue`. That corruption is in the stored data, not in the reconstruction: the
task-contrast columns mark membership with `'x'`, and a `replace('x', True)` somewhere in the
original data preparation caught the name column too, turning `cortex` into `cortetrue`. It
is reproduced verbatim, because these strings are the `region` coordinate labels that every
existing artefact is keyed by — renaming them would quietly fail to line up with anything
already computed. Worth fixing deliberately, across the artefacts at once, rather than here.

Still not recoverable: `data/julia2018_raw/` and `data/julia2018_resting/`, both pre-BIDS
staging directories in no manifest, whose tests now skip.

Older manifests were garbage-collected on the remote; only the 2024-02-29 state survives, so
this is a one-shot recovery.

## Two bugs the missing data had been hiding

Restoring the fixtures made 17 tests run for the first time, and they failed on code rather
than on files.

### Confounds were never applied to the parcellated timeseries

`Parcellation.extract_timeseries` called

```python
ts = self.masker_.fit_transform(img, confound, sample_mask)
```

positionally, against a nilearn signature that is now
`fit_transform(imgs, y=None, confounds=None, sample_mask=None)`. `confound` bound to `y`,
which maskers ignore, and `sample_mask` bound to `confounds`. So confound regression was
silently skipped, and the three subjects that have a non-steady-state volume — AVGP04,
AVGP17NEW, NVGP06 — failed with `Signal length: 125; confound length: 124`, the 124 being the
length of the sample_mask sitting in the confounds slot.

Now passed by keyword. **This is a live-data question, not just a test fix**: any timeseries
regenerated with a nilearn new enough to have `y` in that signature, and without this fix,
would have no confound regression in it.

The existing caches are fine, and this was checked rather than assumed. Regenerating
`dosenbach2010` with the fix in place and comparing against the DVC-tracked
`data/Julia2018/timeseries_dosenbach2010.nc5` gives a maximum absolute difference of 4e-12
across all 32 x 124 x 160 values — floating-point noise. They were built with a nilearn
predating the signature change, when the positional call was still correct. Everything
derived from them, including the agreement analysis in `outputs/agreement/`, stands.

Worth repeating the check on any other cache before reusing it:

```python
old = xr.open_dataset('data/Julia2018/timeseries_<atlas>.nc5', engine='h5netcdf')
new = Parcellation(atlas_name='<atlas>').fit().dataset_      # regenerates
np.abs(old['timeseries'].values - new['timeseries'].values).max()
```

### nilearn 0.14 narrowed `standardize_confounds` to bool

Five parcellation loaders passed `standardize_confounds='zscore_sample'`, which is now a
`TypeError`. `True` is what that string meant. Same class of drift as the `legacy_format`
arguments removed in 0.13.

`Parcellation.cache_dataset` also assumed `derivatives/resting_timeseries/` existed; it now
creates it, rather than failing after doing the expensive part.

## Four more bugs the restored data exposed

Beyond the two above, running the pipeline tests for the first time in a long while turned up:

- **Gordon 2014 region names were off by one.** `gordon.load_masker` passed
  `labels=['0'] + [...]`, prepending a background entry that nilearn 0.14 does not want. Every
  column was reported as the region below it — column 0 as region '0' when it is atlas region
  1 — and nilearn warned "Too many names for the indices" on every call. The prefix is gone.

- **Regions dropped by a subject's brain mask silently misaligned the feature matrix.**
  A `NiftiLabelsMasker` given a per-subject mask drops atlas regions with no voxels under it
  and packs the survivors left, so subjects came back with different widths — 329 to 333 of
  333 for Gordon — and column *k* meant a different part of the brain per participant.
  Stacking them either raised or produced a feature matrix that is quietly wrong.
  `Parcellation._align_regions` now reindexes each subject to the full atlas order.
  nilearn's own `keep_masked_labels=True` is deprecated and removed in 0.15, hence doing it
  explicitly. Atlases using spheres or probabilistic maps never drop anything and are
  untouched; `dosenbach2010` still reproduces the DVC cache exactly.

- **`Parcellation.transform('all')` raised.** Every notebook and test passes the string
  `'all'`, which reached `X.reshape(-1)` and raised `AttributeError`; only `None` worked.
  `fit()` also never set `dataset_`, which `ConnectivityPipeline` and `CerebellumPipeline`
  both expose and the tests expect.

- **`preprocessing/beh.py` broke on pandas 3.** `groupby(...).apply()` no longer passes the
  grouping column into the function, and `events_to_trial` reads `events.trial_index` on its
  first line. Selecting the full column list restores it. This is the module that
  reconstructs trials with response times, so it gates the brain-behaviour analysis in Tier 0
  of the roadmap.

`tests/conftest.py`'s `bids_path` fixture pointed at `data/julia2018` itself, so
`test_beh2bids` rewrote the tracked `sub-*/ses-*/beh/*.tsv` on every run. The values were
right — they agree with the DVC originals to 1e-12 — but pandas now writes shorter float
reprs, so all 61 files came back byte-different and the dataset looked modified after merely
running the tests. The fixture now builds into `tmp_path`.
