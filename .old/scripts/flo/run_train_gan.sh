#!/bin/bash

# Pass GPU ID as arguement, default is 0 if not provided
GPU_ID=${1:-0}

# Experiment save/load directory - remove from python command to allow program to create unique directory
WORK_DIR=experiments/flo/experiment_name

# Configuration file
CONFIG=configs/flo.json

python run.py --config $CONFIG --gpu $GPU_ID --train-gan --work-dir $WORK_DIR
