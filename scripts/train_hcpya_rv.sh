#!/bin/bash
python train.py \
  --cohort hcpya \
  --task rv \
  --x_path data/X_rv_hcpya.npy \
  --y_path data/y_rv_hcpya.npy \
  --groups_path data/groups_hcpya.npy \
  --output_dir results
