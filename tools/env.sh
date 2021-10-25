#!/bin/bash

conda activate pointpillars
ulimit -s 65535
export LD_LIBRARY_PATH=/root/miniconda3/envs/pointpillars/lib/python3.6/site-packages/spconv/:$LD_LIBRARY_PATH
#export PYTHONPATH=/point_pillars/OpenPCDet:$PYTHONPATH
#export LD_PRELOAD=/home/a249s197/miniconda3/envs/pointpillars/lib/python3.6/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
