#!/bin/bash


gen_data()
{
	pushd /point_pillars/nuscenes_dataset/v1.0-mini
	rm -rf gt_database* *pkl
	popd

	pushd /point_pillars/OpenPCDet
	python -m pcdet.datasets.nuscenes.nuscenes_dataset \
		--func create_nuscenes_infos \
		--cfg_file tools/cfgs/dataset_configs/nuscenes_mini_dataset.yaml \
		--version v1.0-mini
	popd
	sleep 1
}


rm -f eval_dict*
# Calibration
str='mini_train, mini_val = mini_val, mini_train'
sed_str_calib='s/#'$str'/'$str'/g'
sed --follow-symlinks -i "$sed_str_calib" splits.py
gen_data
for i in 10 12
do
	./run_tests.sh single $i
done
rm -f eval_dict*

# Test
sed_str_test='s/'$str'/#'$str'/g'
sed --follow-symlinks -i "$sed_str_test" splits.py
gen_data
./run_tests.sh methods $1 $2 $3
