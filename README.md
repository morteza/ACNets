# Attentional Control Networks

ACNets aims to predict action video gaming experience from the resting-state fMRI data and interpret the results. Most of the analysis is implemented in the [notebooks/resting_state](notebooks/resting_state).

It was originally conceived as discovering control-related brain networks using both resting and task fMRI. See the [resources](#additional-resources) section below for details.

## Data

All the raw and derivative datasets are stored in the [data/](data/) folder. Check the folder for a more detailed explanation on how we used HPC and DataLad to manage big datasets of this project.


## Additional Resources

- [ACNets folder on GDrive](https://drive.google.com/drive/folders/1azOq3-tWNipn3vOrgbFzos4cJHOeBZKO?usp=sharing)

- [Initial analysis plan](https://docs.google.com/document/d/17bTvlyH8pX1pIjn28PLyDpQGEmSQ2wki0fiB5TeDuaE/edit?usp=sharing) (Google Docs)

- [Resting state analysis plan](https://docs.google.com/document/d/1gM5IVyKHw9-r9RDRjl158D-yEBbwWnYk1FNUBx_bVic/edit?usp=sharing) (Google Docs)

### Replications

During the initial phases of the project, we used the julia2018 dataset to test and  replicate some of the previous studies, mainly:

- [dosenbach2007 replication notebooks](notebooks/replications/dosenbach2007/)

- task-fMRI [föcker2018 replication notebooks](notebooks/replications/föcker2018/)

They are now obsolete and archived.


## Development

See the [development guide](docs/development.md).

## Conventions

Don't hesitate to use [BIDS](https://bids-specification.readthedocs.io/en/stable/) to name data and folders.

In addition to BIDS, the codebase is aimed to be compatible with the [xCIT style guide and Convention](https://).

## License

This codebase, its project folder on Google Drive, and julia2018 dataset may contain sensitive and confidential materials. Please do not share them without a proper permit.
