# Attentional Control Networks

ACNets aims to use resting-state fMRI to predict the habitual action video gaming experience by analyzing functional connectivity patterns of the brain networks.



 
## Data

You can access all the necessary data through DVC. To download the data, model checkpoints, and outputs from [Uni.lu HPC](https://hpc.uni.lu), please use the following command:

```bash
dvc pull
```

Upon completion, you should see the following folders: `data/`, `models/`, and `outputs/`.


## Development

See the [development guide](docs/development.md).

### Conventions

We highly recommend using [BIDS](https://bids-specification.readthedocs.io/en/stable/) to name data and folders. The codebase is also designed to be compatible with the [Behaverse Data Model](https://behaverse.github.io/data-model/).


## License

Please note that this codebase and related datasets may contain sensitive and confidential materials. Therefore, it is essential not to share them without obtaining the necessary permit.

