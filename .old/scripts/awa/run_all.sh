#!/bin/bash

# Pass GPU ID as arguement, default is 0 if not provided
GPU_ID=${1:-0}

# To specify a particular working directory add --work-dir $WORK_DIR to the python command.
# If none is provided, a unique experiment directory is reated by the program
#WORK_DIR=experiments/awa/experiment_name

# Configuration file
CONFIG=configs/awa.json

# Generate fake features in the specified domains ("unseen", "seen" or "unseen seen")
DOMAIN="unseen seen"

# Number of features to generate per-class for each domain e.g. ("200" or "200 200")
NUM_FEATURES="1200 300"

# Dataset augmentation operation
# "merge": combine fake data with real dataset
# "replace": replace real dataset with fake data
AUG_OP=replace

python run.py --config $CONFIG --gpu $GPU_ID --train-gan --gen-fake --train-cls --domain $DOMAIN --num-features $NUM_FEATURES --aug-op $AUG_OP
