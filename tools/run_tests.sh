#!/bin/bash

if [ -z $1 ]; then
	printf "Give cmd line arg, profile, methods, or slices"
	exit
fi

export CUDA_LAUNCH_BLOCKING=0

PROF_CMD=""
if [ $1 == 'profile' ]; then
	PROF_CMD="nsys profile -w true -f true --stats=true --trace cuda,osrt,cublas,cudnn,mpi,ucx,openacc,openmp"

	# if want to trace stage2 only
	#NUM_SAMPLES=5
	#ARGS="$ARGS -c nvtx --capture-range-end=repeat-shutdown:$NUM_SAMPLES -p RPNstage2@* \
	#	-e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --sampling-frequency=50000 --cuda-memory-usage=true"
fi

# Set correct python executable path
PYTHON_EXE="chrt -r 90 $PROF_CMD python"
CKPT_FILE="../output/kitti_models/pointpillar_imprecise/abc/ckpt/checkpoint_epoch_80.pth"
CFG_FILE="./cfgs/kitti_models/pointpillar_imprecise.yaml"
CMD="$PYTHON_EXE test.py --cfg_file=$CFG_FILE --ckpt $CKPT_FILE --batch_size=1 \
            --workers 0 --eval_tag abc_simple"
#--set "MODEL.DEADLINE_SEC" $2 "MODEL.METHOD" $1

set -x
if [ $1 == 'profile' ]; then
        #export CUDA_LAUNCH_BLOCKING=1
        $CMD --set "MODEL.DEADLINE_SEC" 10.0 "MODEL.METHOD" 5
        #export CUDA_LAUNCH_BLOCKING=0
elif [ $1 == 'methods' ]; then
	mv -f eval_dict_* backup
	OUT_DIR=exp_data_kitti
	mkdir -p $OUT_DIR
	for m in $(seq 5) # baseline 1 2 3 imp-nosclice imp-slice
	do
		for s in $(seq 0.120 -0.030 0.060)
		do
			$CMD --set "MODEL.DEADLINE_SEC" $s "MODEL.METHOD" $m
			# rename the output and move the corresponding directory
			mv -f eval_dict_*.json $OUT_DIR/eval_dict_m"$m"_d"$s".json
		done
	done
elif [ $1 == 'single_slc' ]; then
        $CMD --set "MODEL.DEADLINE_SEC" 10.0 "MODEL.METHOD" 5
elif [ $1 == 'single_noslc' ]; then
        $CMD --set "MODEL.DEADLINE_SEC" 10.0 "MODEL.METHOD" $2
fi

