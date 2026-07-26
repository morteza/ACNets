# ACNets

Machine learning pipelines for predicting action video gaming experience from resting-state fMRI
functional connectivity.

The repository holds two complementary lines of work, previously kept on separate branches and
now merged:

- **`src/acnets/`** is the classic ML pipeline: parcellation, connectivity extraction, network
  aggregation, and cross-validated classification (linear SVM, XGBoost) with SHAP and
  permutation-importance explanations. This is the codebase behind the manuscript.
- **`src/multimodal/`** is the deep learning line: Keras 3 on the PyTorch backend, Lightning
  data modules for Julia2018 and MPI-LEMON, generative and hierarchical temporal models, and
  `pyspi` integration for large-scale pairwise interaction statistics.

See `docs/paper/` for the manuscript plan and the analysis roadmap that ties the two together.

## Setup

```bash
mamba env create -f environment.yml
mamba activate acnets
```

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
pytest
```

Tests live in `tests/` (`test_*.py`) and alongside the sources in `src/` (`*_test.py`). Both
patterns are collected.

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
