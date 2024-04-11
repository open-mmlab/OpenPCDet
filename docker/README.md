# Zendar's OpenPCDet Docker

The Zendar version of this docker file is based on the Cuda 11.6
image made available in the parent repo.

## Build The Image:

From the root of this repository: `make build`

## Run The Image:

From the root of this repository: `make run`


## Notes:

Note that if you would like to use dynamic voxelization, you need to install
the [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) package, which
isn't in the docker at this time.
