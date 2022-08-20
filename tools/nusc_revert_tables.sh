#!/bin/bash
#
pushd ../data/nuscenes/v1.0-mini/v1.0-mini
for fname in 'sample' 'sample_data' 'instance' 'sample_annotation' 'scene'
do
	if [[ -L "$fname.json" ]]; then
        	rm $fname.json
	        mv $fname.json.backup $fname.json
	fi
done
popd
