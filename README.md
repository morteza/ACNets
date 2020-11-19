# Attentional Control Networks

TODO

[Analysis Plan](https://docs.google.com/document/d/17bTvlyH8pX1pIjn28PLyDpQGEmSQ2wki0fiB5TeDuaE/edit?usp=sharing)

## Data

Please see the [`data/` folder](data/) for details on where to put the data.

## Conventions

Don't hesitate to use [BIDS](https://) for naming data and folder structures. In addition to that this code is compatible with [xCIT Styleguide and Convention](https://); feel free to use its vocabulary when you name things.


### Paths
Make sure you use file paths that are relative to the project root, thereby it's always safe to assume project root is your working directory (i.e., what `pwd` would return).

The only exception is paths in the notebooks, in which you need to address resources relative to the notebook path.

## Development

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

2. **Install the data**: ACNets datasets are huge in size, thereby they are managed separately by DataLad. Look into the [data](data/) directory for more detailed explanation and how to install the data that you are interested in.

Once all these requirements are established, you can use the ACNets APIs as exposed in the `acnets` module and demonstrated in the [tests/](tests/) directory. ACNets is best used in tandem with Jupyter notebooks, so look into the notebooks directory for use cases and analyses.
