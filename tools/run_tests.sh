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

# Imprecise model
CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_multihead_imprecise.yaml"
CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_imprecise.pth"

#SECOND CBGS
#CFG_FILE="./cfgs/nuscenes_models/cbgs_second_multihead.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_second_multihead_nds6229_updated.pth"

# PointPillars Single Head
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_singlehead.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_dyn_pp_singlehead/default/ckpt/checkpoint_epoch_20.pth"

#PointPillars Multihead
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_multihead_3br.yaml"
#CKPT_FILE="../output/nuscenes_models/pp_multihead_nds5823_updated.pth"
#             Min        Avrg    95perc  99perc  Max
#End-to-end   121.88     127.65  130.65  131.99  133.54 ms
#--------------average performance-------------
#trans_err:       0.2564
#scale_err:       0.2191
#orient_err:      0.1841
#vel_err:         0.5472
#attr_err:        0.2503
#mAP:     0.5024
#NDS:     0.6055

# Centerpoint-pointpillar
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_centerpoint_nds6070.pth"
#             Min        Avrg    95perc  99perc  Max
#End-to-end   136.60     141.26  143.79  144.84  148.17 ms
#--------------average performance-------------
#trans_err:       0.2484
#scale_err:       0.2414
#orient_err:      0.2774
#vel_err:         0.4299
#attr_err:        0.2241
#mAP:     0.6264
#NDS:     0.6711

# Centerpoint-voxel01
#CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_voxel01_centerpoint_nds_6454.pth"
#             Min        Avrg    95perc  99perc  Max
#End-to-end   169.16     202.05  228.21  238.90  249.01 ms
#--------------average performance-------------
#trans_err:       0.2798
#scale_err:       0.1978
#orient_err:      0.2328
#vel_err:         0.3187
#attr_err:        0.2023
#mAP:     0.6758
#NDS:     0.7147

# Centerpoint-voxel0075
#CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_voxel0075_centerpoint_nds_6648.pth"
#             Min        Avrg    95perc  99perc  Max
#End-to-end   273.41     322.60  356.90  372.49  386.75 ms
#--------------average performance-------------
#trans_err:       0.2190
#scale_err:       0.2236
#orient_err:      0.1877
#vel_err:         0.3065
#attr_err:        0.1980
#mAP:     0.7329
#NDS:     0.7530   


# Baseline models
#CFG_FILE="../output/nuscenes_models/cbgs_pp_multihead_1br/default/cbgs_pp_multihead_1br.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_1br/default/ckpt/checkpoint_epoch_20.pth"
#CFG_FILE="../output/nuscenes_models/cbgs_pp_multihead_2br/default/cbgs_pp_multihead_2br.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_2br/default/ckpt/checkpoint_epoch_20.pth"
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_multihead_3br.yaml"
#CKPT_FILE="../output/nuscenes_models/cbgs_pp_multihead_3br/default/ckpt/checkpoint_epoch_20.pth"
export OMP_NUM_THREADS=2

#TASKSET=""
TASKSET="taskset 0xff"

#DATASET="nuscenes_dataset.yaml"
DATASET="nuscenes_mini_dataset.yaml"
ARG="s/_BASE_CONFIG_: cfgs\/dataset_configs.*$"
ARG=$ARG"/_BASE_CONFIG_: cfgs\/dataset_configs\/$DATASET/g"
sed -i "$ARG" $CFG_FILE

CMD="nice --20 $PROF_CMD $TASKSET python test.py --cfg_file=$CFG_FILE \
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
	prfx="cbgs_dyn_pp_multihead_"
	for model in "1br" "2br" "3br" "imprecise" "imprecise" \
		"imprecise" "imprecise" "imprecise" "imprecise" \
		"imprecise" "imprecise" "imprecise" "imprecise" "imprecise"
	do
		if [ $m == 5 ] || [ $m == 8 ]; then
			# These are not needed
			m=$((m+1))
			continue
		fi
		cfg="$prfx""$model"
		CFG_FILE="./cfgs/nuscenes_models/$cfg.yaml"
		CKPT_FILE="../output/nuscenes_models/$cfg/default/ckpt/checkpoint_epoch_20.pth"
		CMD="nice --20 $TASKSET python test.py --cfg_file=$CFG_FILE \
			--ckpt $CKPT_FILE --batch_size=1 --workers 0"
		ARG="s/_BASE_CONFIG_: cfgs\/dataset_configs.*$"
		ARG=$ARG"/_BASE_CONFIG_: cfgs\/dataset_configs\/$DATASET/g"
		sed -i "$ARG" $CFG_FILE
		for s in $(seq $2 $3 $4)
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
        $CMD --set "MODEL.DEADLINE_SEC" 10.0 "MODEL.METHOD" $2
elif [ $1 == 'single2' ]; then
        $CMD --set "MODEL.DEADLINE_SEC" 0.090 "MODEL.METHOD" $2
fi
