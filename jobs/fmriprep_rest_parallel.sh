#!/bin/bash

#SBATCH --job-name=fmriprep
#SBATCH --array=0-9
#SBATCH --output=/work/projects/acnets/logs/%x_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --partition=batch
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


# Subjects with resting t1w/bold/fmap
SUBJECTS_GROUPS=("AVGP01 AVGP02 AVGP03" "AVGP04 AVGP05 AVGP08" "AVGP10 AVGP12NEW AVGP13" "AVGP13NEW AVGP14 AVGP14NEW" "AVGP16NEW AVGP17NEW AVGP18 AVGP20")
SUBJECTS_GROUPS+=("NVGP01 NVGP03 NVGP04" "NVGP06 NVGP07 NVGP08" "NVGP10 NVGP12 NVGP13" "NVGP14 NVGP15 NVGP16" "NVGP17 NVGP17NEW NVGP19 NVGP19NEW")


# Choose subject based on slurm array id
SUBJECTS=${SUBJECTS_GROUPS[$SLURM_ARRAY_TASK_ID]}
GROUP_NAME=${SUBJECTS//[ ]/_}

JOB_NAME=fmriprep_rest_parallel_${GROUP_NAME}
DATASET=julia2018_datalad_v2020.11.8
RANDOM_SEED=42

PROJECT_DIR=/work/projects/acnets/
INPUT_DIR=${SCRATCH}${DATASET}
OUTPUT_DIR=${PROJECT_DIR}derivatives/$JOB_NAME
TMP_WORK_DIR=${SCRATCH}fmriprep_work_${GROUP_NAME}


echo "Slurm job ID: " $SLURM_JOB_ID
echo "Subjects: " $SUBJECTS
echo "Group Name: " $GROUP_NAME

# enable access to the `module` and `singularity`.
[ -f /etc/profile ] && source /etc/profile
module purge
module load tools/Singularity

# prepare dataset and work dir
if [ ! -d $SCRATCH/$DATASET ]; then
    tar xjf ${PROJECT_DIR}backup/datasets/$DATASET.tar.bz2 -C $SCRATCH
fi

# create temp working dir
mkdir -p $TMP_WORK_DIR
mkdir -p $OUTPUT_DIR


# to avoid unexpected results, only create fmriprep SIF if it does not already exist
if [ ! -f ${SCRATCH}fmriprep_latest.simg ]; then
    singularity build \
        ${SCRATCH}fmriprep_latest.simg \
        docker://nipreps/fmriprep:latest
fi


# run fmriprep
singularity run --cleanenv \
    -B $INPUT_DIR:/inputs \
    -B $OUTPUT_DIR:/outputs \
    -B $SCRATCH \
    $SCRATCH/fmriprep_latest.simg \
    --mem 24GB --n-cpus 8 \
    --fs-license-file $HOME/freesurfer_license.txt \
    --work-dir $TMP_WORK_DIR \
    --notrack \
    --skull-strip-t1w skip \
    --write-graph \
    --random-seed $RANDOM_SEED \
    /inputs /outputs participant \
    --participant-label $SUBJECTS
