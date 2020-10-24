#!/bin/bash

#SBATCH --job-name=fmriprep_rest_single_subject
#SBATCH --output=/work/projects/acnets/logs/fmriprep_%j.out
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
INPUT_DIR=$SCRATCH/$DATASET/bids
OUTPUT_DIR=$PROJECT_DIR/derivatives/$JOB_NAME
TMP_WORK_DIR=${SCRATCH}fmriprep_work

# enable access to the `module` cli (HPC 2019: tools/Singularity/3.6.0)
module purge
module load tools/Singularity


date
echo "Slurm job ID: " $SLURM_JOB_ID
echo "logs: `tail -f /work/projects/acnets/logs/fmriprep_$SLURM_JOB_ID`"

# prepare dataset and work dir
if [ ! -d $SCRATCH/$DATASET ]; then
    tar xjf $PROJECT_DIR/backup/$DATASET.tar.bz2 -C $SCRATCH
fi


# create temp working dir
mkdir -p $TMP_WORK_DIR


#  to avoid unexpected results, only create fmriprep SIF if it does not already exist
if [ ! -f $SCRATCH/fmriprep_latest.simg ]; then
    singularity build \
        $SCRATCH/fmriprep_latest.simg \
        docker://poldracklab/fmriprep:latest
fi


# run fmriprep
singularity run --cleanenv \
    -B $INPUT_DIR:/inputs \
    -B $OUTPUT_DIR:/outputs \
    -B $SCRATCH \
    $SCRATCH/fmriprep_latest.simg \
    --mem 24GB --n-cpus 12 \
    --fs-license-file $HOME/freesurfer_license.txt \
    --work-dir $TMP_WORK_DIR \
    --notrack \
    --skull-strip-t1w skip \
    --write-graph \
    /inputs /outputs participant --participant-label $SUBJ
