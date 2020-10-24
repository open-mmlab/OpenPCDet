#!/usr/bin/env bash

set -x
NGPUS=$1
CFG_FILE=${2}
# PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file ${CFG_FILE}

