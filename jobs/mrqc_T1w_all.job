#!/bin/bash

#SBATCH --job-name=mriqc_T1w_all.job
#SBATCH --output=.out/mriqc_T1w_all.out
#SBATCH --ntasks=1
#SBATCH --mem=8gb
# --partition=gpu
#SBATCH --time=2-00:00
#SBATCH --output=logs/all_on_gpu_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


# enable access to the `module` cli
. /etc/profile.d/lmod.sh


module load tools/EasyBuild
module use $HOME/.local/easybuild/modules/all
module load lang/Miniconda3

conda env create -f environment.yml
conda activate acnets

input=/work/projects/acnets/backup/julia2018_BIDS/
output=/work/projects/acnets/mriqc/mriqc_T1w_all/julia2018_BIDS/
# time singularity exec docker://poldracklab/mriqc:0.15.1 mriqc input output participant --no-sub
singularity exec docker://poldracklab/mriqc:latest mriqc $input $output participant -m T1w --no-sub
