#!/usr/bin/env bash

set -x
NGPUS=$1
CFG=$2
BATCH=$3

python train.py  --cfg_file ${CFG} --batch_size ${BATCH}
