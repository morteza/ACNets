# HPC Jobs

TODO: give an overview in more detail


## Submitting jobs

Run the following on HPC Iris to submit a batch job; it returns the `<job_id>` which will be used during the next steps.

```bash
cd jobs/
sbatch <job>.sh
```


to cancel the job:

```bash
scancel <job_id>
```

to check scheduling details of all your submissions:

```bash
squeue -u `whoami`
```

to see scheduling details of a specific job:

```bash
squeue -j <job_id>
```


## Jobs

Here is a list of all available jobs. See the linked scripts for more detail on what they are supposed to perform.

### fmriprep single subject rest

Performs fmriprep preprocessing on a single subject's resting state images.

> see `fmriprep_single_subject_rest.sh`


### mriqc T1w all

Performs quality checks on resting T1 images of all subjects.

> see `mriqc_t1w_all.sh`

