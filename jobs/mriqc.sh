#!/bin/bash

#SBATCH --job-name=mriqc.job
#SBATCH --output=/work/projects/acnets/logs/mriqc_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu

REPO_DIR = /work/projects/acnets/repositories/acnets

module purge
module load tools/Singularity

INPUT=${REPO_DIR}/data/julia2018/
OUTPUT=${REPO_DIR}/data/julia2018/derivatives/mriqc/

mkdir -p $OUTPUT

singularity exec \
    docker://nipreps/mriqc:latest mriqc \
    $INPUT $OUTPUT participant --no-sub
    # -m T1w

singularity exec \
    docker://nipreps/mriqc:latest mriqc \
    $INPUT $OUTPUT group --no-sub
    # -m T1w
