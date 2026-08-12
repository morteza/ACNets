# ACNets

Machine learning pipelines for predicting action video gaming experience from resting-state fMRI
functional connectivity.

The repository holds two complementary lines of work, previously kept on separate branches and
now merged:

- **`src/acnets/`** is the classic ML pipeline: parcellation, connectivity extraction, network
  aggregation, and cross-validated classification (linear SVM, XGBoost) with SHAP and
  permutation-importance explanations. This is the codebase behind the manuscript.
- **`src/multimodal/`** is the deep learning line: plain PyTorch data modules for Julia2018 and
  MPI-LEMON, and generative and hierarchical temporal models.

See `docs/paper/` for the manuscript plan and the analysis roadmap that ties the two together.

## Setup

[uv](https://docs.astral.sh/uv/) on Python 3.14, pinned in `.python-version`. uv fetches the
interpreter itself, so nothing needs to be installed first.

```bash
uv sync                  # the classic pipeline and the dev tools
uv sync --all-extras     # adds torch, ray, quartodoc and pyinform
```

Then either `uv run <command>` or activate `.venv` directly.

Two things worth knowing about the resolution, both explained in `pyproject.toml`:

- numpy settles on 2.4 rather than the newest, because `shap` needs `numba` and numba's numpy
  ceiling trails numpy by a release or two.
- `transfer_entropy` specifications do not run on an arm64 mac. `pyinform` installs, but the
  `libinform` it vendors is x86_64 only and fails at call time.

## Running the analyses

```bash
uv run python scripts/run_agreement.py             # cross-specification attribution agreement
uv run python scripts/run_agreement.py --n-jobs 1  # serial; results are identical
```

The agreement run is parallel over cross-validation folds and takes about 20 seconds on 12
cores. `--help` lists the grid and cross-validation options.

## Data

Data, model checkpoints, and outputs are versioned with DVC and stored on
[Uni.lu HPC](https://hpc.uni.lu). Datasets under `data/` are tracked per dataset:

```bash
dvc pull                      # everything
dvc pull data/Julia2018.dvc   # AVGP vs NVGP resting-state fMRI
dvc pull data/LEMON.dvc       # MPI-LEMON resting-state fMRI
```

The default remote is `aion-cluster`; `iris` is also configured. See `data/README.md`.

## Testing

```bash
uv run pytest
```

Tests live in `tests/` (`test_*.py`) and alongside the sources in `src/` (`*_test.py`). Both
patterns are collected.

A clean checkout has no data, so about 26 tests fail on missing files. All of what they need
is recoverable from the DVC remote, even though the pointer that tracked it was deleted:

```bash
uv run python scripts/fetch_test_fixtures.py --what atlases        # 34 MB
uv run python scripts/fetch_test_fixtures.py --what resting        # 830 MB
uv run python scripts/fetch_test_fixtures.py --what behaviour      # 4 MB
uv run python scripts/fetch_test_fixtures.py --what dosenbach2007  # derived
```

That gives 225 passed, 3 skipped. The three skips are tests of superseded code whose inputs
were never tracked at all; they skip conditionally, so they run wherever the data exists.
`MERGE_NOTES.md` explains where each fixture came from and how the dosenbach2007 atlas was
reconstructed.

## Development

See the [development guide](docs/development.md).

## License

Please note that this codebase and related datasets may contain sensitive and confidential
materials. Therefore, it is essential not to share them without obtaining the necessary permit.

## Acknowledgments

This research was supported by the Luxembourg National Research Fund
(ATTRACT/2016/ID/11242114/DIGILEARN) and the Swiss National Research Foundation
(100014_178814).

## Citation

> Ansarinia, M., Föcker, J., Lepsien, J., Bavelier, D., Cardoso-Leite, P. (in prep). Cognitive
> control networks connectivity to the sensorimotor network differs across action video game
> players and non-video game players: a rs-fMRI study.
