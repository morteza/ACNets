# Development

You need to take care of the following things before developing ACNets:

1. **Prepare the environment**: ACNets requires Conda to manage dependencies. All the dependencies are listed in the `environment.yml`.

Overall, you run the following commands to create a Conda environment called `acnets`, install all the dependencies, and prepare the development environment:

```bash
conda env create --file environment.yml
conda activate acnets
pip install -e .       # to install acnets package
```

2. **Attach the datasets**: ACNets datasets are huge in size, thereby managed separately by DataLad. Look into the [data](../data/) directory for a more detailed explanation and how to install only a subset of the data that you are interested in.

Once all these requirements are established, you can use the APIs which are exposed in the `acnets` module and demonstrated in the [tests/](../tests/) directory. ACNets is best used in tandem with Jupyter notebooks, so look into the [notebooks directory](../notebooks) for use cases and analyses.

## Paths
Make sure you use file paths that are relative to the project root, thereby it's always safe to assume project root is your working directory (working directory is what `pwd` would return).

The only exception is paths in the notebooks, in which you need to address resources relative to the notebook path. There are however workarounds to solve that. See [running notebooks](running_notebooks.md) for more details.

