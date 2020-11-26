# Development

You need to take care of the following things before developing ACNets:

1. **Prepare development environment**: ACNets requires pip to install dependencies (Python v3.6 and newer). All of the core dependencies are listed in the `requirements.txt`, but there is only one binary dependency that requires other package managers:

- If you have a MacOS computer with Homebrew installed, you can run `brew install dcm2niix`.
- If you prefer Conda, `conda install -c conda-forge dcm2niix` on Linux, MacOS or Windows.

Overall, you can install all the dependencies like this:

```bash
pip install -U -r requirements.txt
pip3 install -e .       # to install acnets package
brew install dcm2niix   # using Homebrew on macOS
# conda install -c conda-forge dcm2niix  # using Conda
```

2. **Install the data**: ACNets datasets are huge in size, thereby they are managed separately by DataLad. Look into the [data](../data/) directory for more detailed explanation and how to install the data that you are interested in.

Once all these requirements are established, you can use the APIs which are exposed in the `acnets` module and demonstrated in the [tests/](../tests/) directory. ACNets is best used in tandem with Jupyter notebooks, so look into the [notebooks directory](../notebooks) for use cases and analyses.

## Paths
Make sure you use file paths that are relative to the project root, thereby it's always safe to assume project root is your working directory (working directory is what `pwd` would return).

The only exception is paths in the notebooks, in which you need to address resources relative to the notebook path. There are however workarounds to solve that. See [running notebooks](running_notebooks.md) for more details.

