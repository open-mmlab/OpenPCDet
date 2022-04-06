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
#
## Calibration
. revert_tables.sh
cp -r 150ms_backup/* .
. copy_new_tables.sh
#rm -f eval_dict*
#str='mini_train, mini_val = mini_val, mini_train'
#sed_str_calib='s/#'$str'/'$str'/g'
#sed --follow-symlinks -i "$sed_str_calib" splits.py
#gen_data
#for i in 8 7 6 5 4
#do
#	./run_tests.sh single $i
#done
#
# Gen test data
rm -f eval_dict*
sed_str_test='s/'$str'/#'$str'/g'
sed --follow-symlinks -i "$sed_str_test" splits.py
gen_data

# Test 150 ms
./run_tests.sh methods 0.140 -0.010 0.110

# Test 100 ms
. revert_tables.sh
cp -r 100ms_backup/* .
. copy_new_tables.sh
gen_data
./run_tests.sh methods 0.100 -0.010 0.050
