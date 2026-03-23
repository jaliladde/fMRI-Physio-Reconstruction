#!/bin/bash
python train.py \
  --cohort hcpa \
  --task hrv \
  --x_path data/X_hrv_hcpa.npy \
  --y_path data/y_hrv_hcpa.npy \
  --groups_path data/groups_hcpa.npy \
  --output_dir results
