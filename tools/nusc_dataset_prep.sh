#!/bin/bash

for period in "100" "150"
do
	TABLES_PATH="nusc_tables_and_dicts/$period"
	mkdir -p "$TABLES_PATH/tables"
	python nusc_dataset_utils.py populate_annos $period
	mv -f sample.json sample_data.json instance.json sample_annotation.json scene.json "$TABLES_PATH/tables"
	. nusc_link_tables.sh "$TABLES_PATH/tables"
	python nusc_dataset_utils.py generate_dicts
	mv -f token_to_anns.json token_to_pos.json $TABLES_PATH/../
	. nusc_revert_tables.sh
done
