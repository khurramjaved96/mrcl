#!/bin/bash
#SBATCH --account=def-whitem
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --mem=30000M               # memory per node
#SBATCH --time=0-12:59            # time (DD-HH:MM)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cp ~/data/omni.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/omni.zip  -d $SLURM_TMPDIR/
mv $SLURM_TMPDIR/omni2 $SLURM_TMPDIR/omni
unzip $SLURM_TMPDIR/omni/omniglot-py/images_background.zip -d $SLURM_TMPDIR/omni/omniglot-py/  >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt
unzip $SLURM_TMPDIR/omni/omniglot-py/images_evaluation.zip -d $SLURM_TMPDIR/omni/omniglot-py/  >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt
python stationary.py --rln 6 --update_lr 0.03 --name Final_Fixed_0_03_large_er --meta_lr 1e-6 --update_step 5 --steps 1000000 --dataset-path $SLURM_TMPDIR/omni/
