#!/bin/bash

#SBATCH --job-name=mriqc_t1w.job
#SBATCH --output=/work/projects/acnets/logs/slurm_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu

module purge
module load tools/Singularity

input=/work/projects/acnets/backup/julia2018/bids/
output=/work/projects/acnets/derivatives/mriqc/julia2018_bids/
# time singularity exec docker://poldracklab/mriqc:0.15.1 mriqc input output participant --no-sub
singularity exec \
    docker://poldracklab/mriqc:latest mriqc \
    $input $output participant \
    -m T1w --no-sub
