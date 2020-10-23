#!/bin/bash

#SBATCH --job-name=fmriprep_rest_single_subject
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24gb
#SBATCH --partition=bigmem
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


# job parameters
SUBJ=AVGP01
DATASET=julia2018_datalad_v2020.10.1
JOB_NAME=fmriprep_rest_$SUBJ
PROJECT_DIR=/work/projects/acnets
INPUT_DIR=$SCRATCH/$DATASET
OUTPUT_DIR=$PROJECT_DIR/derivatives/$JOB_NAME


# enable access to the `module` cli (HPC 2019: tools/Singularity/3.6.0)
module purge
module load tools/Singularity/3.6.0


echo "Slurm job ID: " $SLURM_JOB_ID
date


# temp work dir to keep intermediate stuff
mkdir -p $SCRATCH/work


# extract dataset
tar xjf $PROJECT_DIR/backup/$DATASET.tar.bz2 -C $SCRATCH


# create singularity image
singularity build \
    $PROJECT_DIR/singularity_images/fmriprep_latest.simg \
    docker://poldracklab/fmriprep:latest


# run fmriprep
singularity run --cleanenv \
    --bind $SCRATCH/work \
    $PROJECT_DIR/singularity_images/fmriprep_latest.simg \
    $INPUT_DIR $OUTPUT_DIR participant \
    --mem 24GB --n-cpus 12 \
    --work-dir $SCRATCH/work \
    --fs-license-file $HOME/freesurfer_license.txt \
    --notrack \
    --skull-strip-t1w skip \
    --write-graph \
    --participant-label $SUBJ
