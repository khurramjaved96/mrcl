#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=30000M               # memory per node
#SBATCH --time=0-12:59            # time (DD-HH:MM)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp ~/imagenet.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/imagenet.zip -d $SLURM_TMPDIR/ >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt
python mrcl_imagenet.py --dataset-path $SLURM_TMPDIR/imagenet/ --name IMAGENET_0_1_traj_10 --update_lr 0.1