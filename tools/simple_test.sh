#!/bin/bash

CKPT_FILE="../output/kitti_models/pointpillar_imprecise/abc/ckpt/checkpoint_epoch_80.pth"
CFG_FILE="./cfgs/kitti_models/pointpillar_imprecise.yaml"

do_test()
{
	if [ -z $PROFILE ]
	then
		PROF=""
	else
		PROF="nsys profile -w true -f true --stats=true --trace cuda,osrt,cublas,cudnn,mpi,ucx,openacc,openmp"
	fi
        chrt -r 90 $PROF python test.py  --cfg_file=$CFG_FILE --ckpt $CKPT_FILE --batch_size=1 \
            --workers 0 --eval_tag "abc_simple" --set "MODEL.DEADLINE_SEC" $2 "MODEL.METHOD" $1
}

#LOGFILE=log_simple_test.txt
#rm -f $LOGFILE && touch $LOGFILE
do_test $1 $2
