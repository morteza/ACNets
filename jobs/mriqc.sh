#!/bin/bash

#SBATCH --job-name=mriqc_t1w.job
#SBATCH --output=/work/projects/acnets/logs/mriqc_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu

module purge
module load tools/Singularity

input=/work/projects/acnets/backup/julia2018/
output=/work/projects/acnets/derivatives/mriqc/julia2018/

singularity exec \
    docker://nipreps/mriqc:latest mriqc \
    $input $output participant --no-sub
    # -m T1w

singularity exec \
    docker://nipreps/mriqc:latest mriqc \
    $input $output group --no-sub
    # -m T1w
