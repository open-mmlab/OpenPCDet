#!/bin/bash

TABLES_PATH=$(realpath $1)
pushd ../data/nuscenes/v1.0-mini/v1.0-mini
for fname in 'sample' 'sample_data' 'instance' 'sample_annotation' 'scene'
do
	if [[ ! -L "$fname.json" ]]; then
	        mv $fname.json $fname.json.backup # backup the original tables
        	ln -s "$TABLES_PATH/$fname.json"
	fi
done
popd
