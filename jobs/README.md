# HPC Jobs

TODO: give an overview


## Submitting jobs

The following code submits a job to HPC and returns the `<job_id>`, which will be used in the next steps.

```bash
si

cd /work/project/acnets/repositories
git clone ssh://git@gitlab.uni.lu:8022/xcit/brain-imaging/acnets.git & cd acnets/

cd data/
datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~julia2018
cd julia2018/
datalad get ./*
datalad unlock ./*

cd /work/project/acnets/repositories/acnets/
sbatch jobs/<job>.sh
```

to track logs:

```
tail -f /work/projects/acnets/logs/slurm_<job_id>.out
```

to see expected start time of your submissions:

```bash
squeue --start -u $USER
```

to cancel the job:

```bash
scancel <job_id>
```

to check scheduling details of all your submissions:

```bash
squeue -u $USER
```

to see scheduling details of a specific job:

```bash
squeue -j <job_id>
```


## Jobs

Here is a list of all available jobs. See the linked scripts for more detail on what they are supposed to perform.

### `fmriprep_single_subject`

Performs fmriprep preprocessing on a single subject's resting state images.

> see `fmriprep_single_subject.sh`


### `mriqc_t1w`

Performs quality checks on resting T1 images of all subjects.

> see `mriqc_t1w.sh`

