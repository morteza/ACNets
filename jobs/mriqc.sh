#!/bin/bash

#SBATCH --job-name=mriqc.job
#SBATCH --output=/work/projects/acnets/logs/mriqc_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


module purge
module load tools/Singularity

REPO_DIR="/work/projects/acnets/repositories/acnets"
BIDS_DIR="${REPO_DIR}/data/julia2018/"
OUTPUT_DIR="${REPO_DIR}/data/julia2018/derivatives/mriqc/"

mkdir -p $OUTPUT_DIR

singularity run \
    --bind $BIDS_DIR:/bids_dir \
    --bind $OUTPUT_DIR:/output_dir \
    docker://nipreps/mriqc:latest \
    /bids_dir /output_dir participant --no-sub
    # -m T1w

singularity run \
    --bind $BIDS_DIR:/bids_dir \
    --bind $OUTPUT_DIR:/output_dir \
    docker://nipreps/mriqc:latest \
    /bids_dir /output_dir group --no-sub
    # -m T1w
