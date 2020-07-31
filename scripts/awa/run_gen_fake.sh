#!/bin/bash

# Pass GPU ID as arguement, default is 0 if not provided
GPU_ID=${1:-0}

# Experiment save/load directory
WORK_DIR=experiments/awa/experiment_name

# Configuration file
CONFIG=configs/awa.json

# Generate fake features in the specified domains ("unseen", "seen" or "unseen seen")
DOMAIN="unseen seen"

# Number of features to generate per-class for each domain e.g. ("200" or "200 200")
NUM_FEATURES="1200 300"

python run.py --config $CONFIG --gpu $GPU_ID --gen-fake --domain $DOMAIN --num-features $NUM_FEATURES --work-dir $WORK_DIR
