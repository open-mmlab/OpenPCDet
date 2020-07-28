## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1)
* CUDA 9.0 or higher
* `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634))


### Install `pcdet v0.3`
NOTE: Please re-install `pcdet v0.3` by running `python setup.py develop` even if you have already installed previous version.

a. Clone this repository.
```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the non-official implementation from [`spconv`](https://github.com/traveller59/spconv). 
Note that we use the initial version of `spconv`, make sure you install the `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.

c. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```
