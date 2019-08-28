#!/usr/bin/env bash

cp ~/data/omni.zip $SLURM_TMPDIR/
unzip $SLURM_TMPDIR/omni.zip  -d $SLURM_TMPDIR/
mv $SLURM_TMPDIR/omni2 $SLURM_TMPDIR/omni
unzip $SLURM_TMPDIR/omni/omniglot-py/images_background.zip -d $SLURM_TMPDIR/omni/omniglot-py/  >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt
unzip $SLURM_TMPDIR/omni/omniglot-py/images_evaluation.zip -d $SLURM_TMPDIR/omni/omniglot-py/  >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt
python mrcl_classification_new_protocol.py --rln 6 --update_lr 0.03 --name mrcl_omniglot_NEW_PROTOCOL --update_step 5 --steps 100000 --dataset-path $SLURM_TMPDIR/omni/