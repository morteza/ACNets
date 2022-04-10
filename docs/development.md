# Development

Before running notebooks or accessing the `acnets` package, you need to take care of a few things:

1. **Prepare the environment**: ACNets requires Conda to manage dependencies. All the dependencies are listed in the `environment.yml`.

You need to create a Conda environment called `acnets` and install all the dependencies:

```bash
conda env create --file environment.yml
conda activate acnets
pip install -e .       # to install the `acnets` package
```

2. **Attach the datasets**: ACNets datasets are huge in size, thereby managed separately by DataLad. Look into the [data](../data/) directory for a more detailed explanation. To save space, install only the subset of the data that you are interested in.


Once all these requirements are in place, APIs are exposed via the `acnets` module as demonstrated in the [tests/](../tests/) directory. ACNets is best used in tandem with Jupyter notebooks, so look into the [notebooks directory](../notebooks) for different use cases.

## Paths
Make sure you use file paths that are relative to the project root, so it will be always safe to assume project root is your working directory (as provided by the `pwd` command).

The only exception is when you are developing notebooks in tools other than VSCode. In that case you need to address resources relative to the notebook path. There are however workarounds to solve that. See [running notebooks](running_notebooks.md) for possible solutions.
