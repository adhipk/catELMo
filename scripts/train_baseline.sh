#!/bin/bash
#SBATCH -p general
#SBATCH -t 5-00:0:00
#SBATCH --mem=100GB
#SBATCH -G a100:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sshah205@asu.edu
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate bap
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1 python -W ignore ../bap.py \
                --embedding catELMo \
                --split epitope \
                --gpu 0 \
                --fraction 1 \
                --seed 42 \