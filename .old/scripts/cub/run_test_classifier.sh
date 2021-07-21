#!/bin/bash

# Pass GPU ID as arguement, default is 0 if not provided
GPU_ID=${1:-0}

# Experiment save/load directory
WORK_DIR=experiments/cub/experiment_name

# Configuration file
CONFIG=configs/cub.json

python run.py --config $CONFIG --gpu $GPU_ID --test-cls --work-dir $WORK_DIR
