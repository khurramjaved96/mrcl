#!/usr/bin/env bash

mkdir $SLURM_TMPDIR/celeba/
cp -r ~/data/celebb/celeba/ $SLURM_TMPDIR/celeba/
unzip $SLURM_TMPDIR/celeba/celeba/img_align_celeba.zip -d $SLURM_TMPDIR/celeba/celeba >  $SLURM_TMPDIR/log.txt 2> $SLURM_TMPDIR/err.txt