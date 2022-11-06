

#
<br>


- python -m torch.distributed.launch 
- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md

> then collectively synchronizing gradients using the AllReduce primitive. In HPC terminology, this model of execution is called Single Program Multiple Data or SPMD since the same application runs on all application but each one operates on different portions of the training dataset.

#
<br>

- https://discuss.pytorch.org/t/the-right-way-to-distribute-the-training-over-multiple-gpus-and-nodes/29277

