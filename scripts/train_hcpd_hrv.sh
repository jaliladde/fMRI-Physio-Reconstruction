#!/bin/bash
python train.py \
  --cohort hcpd \
  --task hrv \
  --x_path data/X_hrv_hcpd.npy \
  --y_path data/y_hrv_hcpd.npy \
  --groups_path data/groups_hcpd.npy \
  --output_dir results
