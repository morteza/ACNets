#!/bin/bash

#SBATCH --job-name=fmriprep
#SBATCH --output=/work/projects/acnets/logs/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --partition=bigmem
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


# job parameters
JOB_NAME=fmriprep_rest
DATASET=julia2018
RANDOM_SEED=42

PROJECT_DIR=/work/projects/acnets/repositories/acnets
BIDS_DIR=${PROJECT_DIR}/data/${DATASET}
OUTPUT_DIR=${BIDS_DIR}/derivatives/$JOB_NAME
TMP_WORK_DIR=${SCRATCH}fmriprep_work


# Subjects with resting bold and t1w
SUBJECTS_WITH_REST_BOLD="AVGP01 AVGP02 AVGP03 AVGP04 AVGP05 AVGP08 AVGP10 AVGP12NEW AVGP13 AVGP13NEW AVGP14 AVGP14NEW AVGP16NEW AVGP17NEW AVGP18 AVGP20"
SUBJECTS_WITH_REST_BOLD+=" NVGP01 NVGP03 NVGP04 NVGP06 NVGP07 NVGP08 NVGP10 NVGP12 NVGP13 NVGP14 NVGP15 NVGP16 NVGP17 NVGP17NEW NVGP19 NVGP19NEW"


# enable access to the `module` and `singularity`.
[ -f /etc/profile ] && source /etc/profile
module purge
module load tools/Singularity


date
echo "Slurm job ID: " $SLURM_JOB_ID


# create temp working dir
mkdir -p $TMP_WORK_DIR
mkdir -p $OUTPUT_DIR

# run fmriprep
singularity run --cleanenv \
    --bind $BIDS_DIR:/inputs \
    --bind $OUTPUT_DIR:/outputs \
    --bind $SCRATCH \
    docker://nipreps/mriqc:latest \
    --mem 32GB --n-cpus 16 --omp-nthreads 16 \
    --fs-license-file $HOME/freesurfer_license.txt \
    --work-dir $TMP_WORK_DIR \
    --notrack \
    --skull-strip-t1w skip \
    --write-graph \
    --random-seed $RANDOM_SEED \
    /inputs /outputs participant \
    --participant-label $SUBJECTS_WITH_REST_BOLD
