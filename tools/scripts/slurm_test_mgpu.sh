#!/usr/bin/env bash

set -x

PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$GPUS
PY_ARGS=${@:3}
JOB_NAME=eval
SRUN_ARGS=${SRUN_ARGS:-""}

PORT=$(( ( RANDOM % 10000 )  + 10000 ))

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u test.py --launcher slurm ${PY_ARGS} --tcp_port $PORT

