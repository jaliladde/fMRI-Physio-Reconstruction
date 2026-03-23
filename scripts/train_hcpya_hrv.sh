#!/bin/bash
python train.py \
  --cohort hcpya \
  --task hrv \
  --x_path data/X_hrv_hcpya.npy \
  --y_path data/y_hrv_hcpya.npy \
  --groups_path data/groups_hcpya.npy \
  --output_dir results
