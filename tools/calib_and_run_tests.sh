#!/bin/bash

gen_data()
{
	pushd ../data/nuscenes/v1.0-mini
	rm -rf gt_database* *pkl
	popd

	pushd ..
	python -m pcdet.datasets.nuscenes.nuscenes_dataset \
		--func create_nuscenes_infos \
		--cfg_file tools/cfgs/dataset_configs/nuscenes_mini_dataset.yaml \
		--version v1.0-mini
	popd
	sleep 1
}

link_tables_and_dicts()
{
	. nusc_revert_tables.sh
	. nusc_link_tables.sh $1/tables
	for f in token_to_anns.json token_to_pos.json
	do
		rm -f $f
		ln -s $1/$f
	done
}

# Calibration, do it only for 150ms, and use it for all periods
link_tables_and_dicts "nusc_tables_and_dicts/150"

# Gen calib data 150ms
str='mini_train, mini_val = mini_val, mini_train'
sed_str_calib='s/#'$str'/'$str'/g'
sed --follow-symlinks -i "$sed_str_calib" splits.py # calib
gen_data
##CALIBRATION START###
for m in $(seq 4 14)
do
	if [ $m == 5 ] || [ $m == 8 ]; then
		# These are not needed
		m=$((m+1))
		continue
	fi
	# do calibration ignoring deadline
	./run_tests.sh single $m
done
rm -f eval_dict* # clear up
#CALIBRATION END###

# Gen test data 150ms
sed_str_test='s/'$str'/#'$str'/g'
sed --follow-symlinks -i "$sed_str_test" splits.py # test
gen_data

# Test 150 ms
./run_tests.sh methods 0.140 -0.010 0.110

# Gen test data 100ms
link_tables_and_dicts "nusc_tables_and_dicts/100"
gen_data

# Test 100 ms
./run_tests.sh methods 0.100 -0.010 0.050

#Plot
for s in 0 1 2 3
do
	python3 log_plotter.py exp_data_nsc/ $s
done
