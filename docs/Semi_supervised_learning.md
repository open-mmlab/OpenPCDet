# Semi-supervised learning

##  1. Parse the instruction doc of scenes.
tools/generate_pseudo_label/parse_scene_instruction_doc.py
```shell script
python parse_scene_instruction_doc.py
```` 

## 2. Generate pseudo labels with pre-trained pv_rcnn and unlabeled points.
```shell script
python demo.py --cfg_file ${CONFIG_FILE} --ckpt ${ckpt_FILE} --data_path ${DATA_PATH} --output_path ${OUTPUT_PATH}
```

## 3. Generate training infos and ground truth database.
tools/semi_supervised_train.py
```shell script
python semi_supervised_train.py
```

## 4. Train with labeled and pseudo datasets in multi GPUS.
```shell script
bash scripts/dist_train.sh ${GPU_NUM} --cfg_file ${CONFIG_FILE}
```