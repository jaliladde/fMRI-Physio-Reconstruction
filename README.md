# ML Physio Reconstruction - Scripts

This repository contains scripts to train CNN+GRU models for physiological signal reconstruction from fMRI.

## Quick Start

Run one of the scripts:

```bash
bash scripts/train_hcpya_rv.sh
```

## Available Scripts

### RV Models
- scripts/train_hcpd_rv.sh
- scripts/train_hcpya_rv.sh
- scripts/train_hcpa_rv.sh

### HRV Models
- scripts/train_hcpd_hrv.sh
- scripts/train_hcpya_hrv.sh
- scripts/train_hcpa_hrv.sh

## Input Requirements

Each script expects:

- X: (N, 65, features)
- y: (N,)
- groups: (N,) → subject IDs

## Notes

- RV uses 636 features (BOLD + motion)
- HRV uses 630 features (BOLD only)
- Models are trained per age group (HCP-D, HCP-YA, HCP-A)
