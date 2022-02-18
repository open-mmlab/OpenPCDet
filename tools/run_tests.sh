#!/bin/bash

if [ -z $1 ]; then
	printf "Give cmd line arg, profile, methods, or slices"
	exit
fi

export CUDA_LAUNCH_BLOCKING=0

PROF_CMD=""
if [ $1 == 'profile' ]; then
	PROF_CMD="nsys profile -w true \
		--trace cuda,osrt,cublas,nvtx,cudnn \
		--sampling-trigger=timer,sched,cuda \
		--process-scope=process-tree"

	# if want to trace stage2 only
	#NUM_SAMPLES=5
	#ARGS="$ARGS -c nvtx \
	#	--capture-range-end=repeat-shutdown:$NUM_SAMPLES \
	#	-p RPNstage2@* \
	#	-e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	#	--sampling-frequency=50000 --cuda-memory-usage=true"
fi

#CKPT_FILE="../output/kitti_models/pointpillar_imprecise/abc/ckpt/checkpoint_epoch_80.pth"
#CFG_FILE="./cfgs/kitti_models/pointpillar_imprecise.yaml"

#CKPT_FILE="../output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth"
#CFG_FILE="./cfgs/kitti_models/pointpillar.yaml"

#CFG_FILE="./cfgs/nuscenes_models/cbgs_pp_multihead_imprecise_caronly.yaml"

# Imprecise model
CFG_FILE="./cfgs/nuscenes_models/cbgs_pp_multihead_imprecise.yaml"
CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_imprecise/default/ckpt/checkpoint_epoch_20.pth"

# Baseline models
#CFG_FILE="../output/nuscenes_models/cbgs_pp_multihead_1br/default/cbgs_pp_multihead_1br.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_1br/default/ckpt/checkpoint_epoch_20.pth"
#CFG_FILE="../output/nuscenes_models/cbgs_pp_multihead_2br/default/cbgs_pp_multihead_2br.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_2br/default/ckpt/checkpoint_epoch_20.pth"
#CFG_FILE="../output/nuscenes_models/cbgs_pp_multihead_3br/default/cbgs_pp_multihead_3br.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_3br/default/ckpt/checkpoint_epoch_20.pth"

#TASKSET=""
TASKSET="taskset 0xc0"

#DATASET="nuscenes_dataset.yaml"
DATASET="nuscenes_mini_dataset.yaml"
ARG="s/_BASE_CONFIG_: cfgs\/dataset_configs.*$"
ARG=$ARG"/_BASE_CONFIG_: cfgs\/dataset_configs\/$DATASET/g"
sed -i "$ARG" $CFG_FILE

CMD="chrt -r 90 $PROF_CMD $TASKSET python test.py --cfg_file=$CFG_FILE \
	--ckpt $CKPT_FILE --batch_size=1 --workers 0"

set -x
if [ $1 == 'profile' ]; then
        #export CUDA_LAUNCH_BLOCKING=1
        $CMD --set "MODEL.DEADLINE_SEC" 10.0 "MODEL.METHOD" 3
        #export CUDA_LAUNCH_BLOCKING=0
elif [ $1 == 'methods' ]; then
	mv -f eval_dict_* backup
	OUT_DIR=exp_data_nsc
	mkdir -p $OUT_DIR
	m=1
	prfx="cbgs_pp_multihead_"
	for model in "1br" "2br" "3br" "imprecise" "imprecise" "imprecise" "imprecise"
	do
		cfg="$prfx""$model"
		CFG_FILE="./cfgs/nuscenes_models/$cfg.yaml"
		CKPT_FILE="../output/nuscenes_models/$cfg/default/ckpt/checkpoint_epoch_20.pth"
		CMD="chrt -r 90 $TASKSET python test.py --cfg_file=$CFG_FILE \
			--ckpt $CKPT_FILE --batch_size=1 --workers 0"
		ARG="s/_BASE_CONFIG_: cfgs\/dataset_configs.*$"
		ARG=$ARG"/_BASE_CONFIG_: cfgs\/dataset_configs\/$DATASET/g"
		sed -i "$ARG" $CFG_FILE
		for s in $(seq 0.140 -0.010 0.100)
		do
			OUT_FILE=$OUT_DIR/eval_dict_m"$m"_d"$s".json
			if [ -f $OUT_FILE ]; then
				printf "Skipping $OUT_FILE test.\n"
			else
				$CMD --set "MODEL.DEADLINE_SEC" $s "MODEL.METHOD" $m
				# rename the output and move the corresponding directory
				mv -f eval_dict_*.json $OUT_DIR/eval_dict_m"$m"_d"$s".json
			fi
		done
		m=$((m+1))
	done
elif [ $1 == 'single' ]; then
        $CMD --set "MODEL.DEADLINE_SEC" 0.100 "MODEL.METHOD" $2
fi
