# Development

## Conventions

We highly recommend using [BIDS](https://bids-specification.readthedocs.io/en/stable/) to name data and folders. The codebase is also designed to be compatible with the [Behaverse Data Model](https://behaverse.github.io/data-model/).


# Setup

Before running notebooks, you need to take care of a few things:


1. **Prepare the environment**: ACNets uses [uv](https://docs.astral.sh/uv/), which fetches the
   Python interpreter itself (3.14, pinned in `.python-version`), so nothing needs installing
   first:

```bash
uv sync                  # the classic pipeline and the dev tools
uv sync --all-extras     # adds torch, ray, quartodoc and pyinform
```

   Run things with `uv run <command>`, or activate `.venv` directly. Conda, mamba,
   `environment.yml` and pixi were all used at various points and are all gone;
   `pyproject.toml` is the only place dependencies are declared.

2. **Attach the datasets**: ACNets datasets are huge in size, thereby managed separately by DVC. Use the following command to download the datasets:

```bash
dvc pull
```

ACNets is best used in tandem with Jupyter notebooks, so look into the [notebooks directory](../notebooks) for different use cases.

## Paths

Make sure you use file paths that are relative to the project root, so it will be always safe to assume project root is your working directory (as provided by the `pwd` command).

The only exception is when you are developing notebooks in tools other than VSCode. In that case you need to address resources relative to the notebook path. There are however workarounds to solve that. See [notebooks](notebooks.md) for possible solutions.
