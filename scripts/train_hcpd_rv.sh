#!/bin/bash
python train.py \
  --cohort hcpd \
  --task rv \
  --x_path data/X_rv_hcpd.npy \
  --y_path data/y_rv_hcpd.npy \
  --groups_path data/groups_hcpd.npy \
  --output_dir results
