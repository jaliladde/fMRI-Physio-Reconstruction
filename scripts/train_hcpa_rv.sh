#!/bin/bash
python train.py \
  --cohort hcpa \
  --task rv \
  --x_path data/X_rv_hcpa.npy \
  --y_path data/y_rv_hcpa.npy \
  --groups_path data/groups_hcpa.npy \
  --output_dir results
