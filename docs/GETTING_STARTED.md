## Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](tools/cfgs) for different datasets, like [tools/cfgs/kitti_models/](tools/cfgs/kitti_models/). 
 


### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
Note that the `--batch_size` depends on the number of your training GPUs, 
please refer to `Model Zoo` of [README.md](../README.md) for the setting of batch_size for different models.  

* Train with multiple GPUs:
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}  --epochs 80
```

* Train with multiple machines:
```shell script
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} \ 
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs 80
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs 50
```
