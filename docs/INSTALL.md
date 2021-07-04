# Installation

### Requirements
The codes are tested in the following environment:
* Ubuntu 18.04
* Python 3.6
* PyTorch 1.5
* CUDA 10.1
* OpenPCDet v0.3.0
* spconv v1.2.1

### Install ONCE benchmark

a. Clone this repository.
```shell
git clone https://github.com/JiagengMao/ONCE_Benchmark.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 

c. Compile CUDA operators by running the following command:
```shell
python setup.py develop
```

d. Compile DCN (Deformable Convs). This step is optional if you don't use DCN in CenterPoints. You may meet some errors if dcn is not compiled, but that's ok. You can simple delete anything related to .dcn kernel if it is not needed.
```shell
cd pcdet/ops/dcn
python setup.py develop
```