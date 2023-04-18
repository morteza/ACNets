# Development

## Conventions

We highly recommend using [BIDS](https://bids-specification.readthedocs.io/en/stable/) to name data and folders. The codebase is also designed to be compatible with the [Behaverse Data Model](https://behaverse.github.io/data-model/).


# Setup

Before running notebooks, you need to take care of a few things:


1. **Prepare the environment**: ACNets requires Conda/Mamba to manage dependencies. Run the following code to create an environment called `acnets` and install all the dependencies in it:

```bash
mamba env create -f environment.yml
mamba activate acnets
```

2. **Attach the datasets**: ACNets datasets are huge in size, thereby managed separately by DVC. Use the following command to download the datasets:

```
dvc pull
```

ACNets is best used in tandem with Jupyter notebooks, so look into the [notebooks directory](../notebooks) for different use cases.

## Paths

Make sure you use file paths that are relative to the project root, so it will be always safe to assume project root is your working directory (as provided by the `pwd` command).

The only exception is when you are developing notebooks in tools other than VSCode. In that case you need to address resources relative to the notebook path. There are however workarounds to solve that. See [running notebooks](running_notebooks.md) for possible solutions.
