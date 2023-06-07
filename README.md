# ACNets

This code implements a machine learning pipeline for predicting the action video gaming experience from resting-state fMRI functional connectivity.

## Setup

To install the required dependencies, please run the following command:

```bash
mamba env create -f environment.yml
mamba activate acnets
```

 
## Data

You can access all the data using DVC. To download the data, model checkpoints, and outputs from [Uni.lu HPC](https://hpc.uni.lu), please use the following command:

```bash
dvc pull
```

Upon completion, you should see the following folders: `data/`, `models/`, and `outputs/`.


## Development

See the [development guide](docs/development.md).


## License

Please note that this codebase and related datasets may contain sensitive and confidential materials. Therefore, it is essential not to share them without obtaining the necessary permit.


## Acknowledgments

This research was supported by the Luxembourg National Research Fund (ATTRACT/2016/ID/11242114/DIGILEARN) and the Swiss National Research Foundation (100014_178814).


## Citation


> Ansarinia, M., Föcker, J., Lepsien, J., Bavelier, D., Cardoso-Leite, P. (in prep). Cognitive control networks connectivity to the sensorimotor network differs across action video game players and non-video game players: a rs-fMRI study.

