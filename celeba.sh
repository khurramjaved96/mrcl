#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=30000M               # memory per node
#SBATCH --time=1-12:59            # time (DD-HH:MM)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir $SLURM_TMPDIR/celeba/
cp -r ~/data/celebb/celeba/ $SLURM_TMPDIR/celeba/
unzip $SLURM_TMPDIR/celeba/celeba/img_align_celeba.zip -d $SLURM_TMPDIR/celeba/celeba >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt
