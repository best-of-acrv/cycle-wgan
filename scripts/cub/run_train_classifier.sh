#!/bin/bash

# Pass GPU ID as arguement, default is 0 if not provided
GPU_ID=${1:-0}

# Experiment save/load directory
WORK_DIR=experiments/cub/experiment_name

# Configuration file
CONFIG=configs/cub.json

# Dataset augmentation operation
# "merge": combine fake data with real dataset
# "replace": replace real dataset with fake data
AUG_OP=merge

python run.py --config $CONFIG --gpu $GPU_ID --train-cls --aug-op $AUG_OP --work-dir $WORK_DIR
