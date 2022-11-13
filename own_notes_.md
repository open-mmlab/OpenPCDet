

- https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md

#
<br>

##### SPCONV INstall and checks for previous versions 

- https://github.com/traveller59/spconv#install

> Install - You need to install python >= 3.7 first to use spconv 2.x.
You need to install CUDA toolkit first before using prebuilt binaries or build from source.
You need at least CUDA 11.0 to build and run spconv 2.x. We won't offer any support for CUDA < 11.0.

#
#
<br>

- pip install spconv-cu117

#

```bash
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ pip install spconv-cu117
Collecting spconv-cu117
  Downloading spconv_cu117-2.2.6-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (69.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 69.5/69.5 MB 7.4 MB/s eta 0:00:00
Collecting ccimport>=0.4.0
  Downloading ccimport-0.4.2-py3-none-any.whl (27 kB)
Collecting pybind11>=2.6.0
  Downloading pybind11-2.10.1-py3-none-any.whl (216 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 216.1/216.1 kB 4.5 MB/s eta 0:00:00
Collecting cumm-cu117>=0.3.7
  Downloading cumm_cu117-0.3.7-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.2/23.2 MB 14.6 MB/s eta 0:00:00
Requirement already satisfied: numpy in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from spconv-cu117) (1.23.4)
Collecting pccm>=0.4.0
  Downloading pccm-0.4.4-py3-none-any.whl (68 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68.8/68.8 kB 1.5 MB/s eta 0:00:00
Collecting fire
  Using cached fire-0.4.0-py2.py3-none-any.whl
Collecting ninja
  Downloading ninja-1.11.1-py2.py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (145 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 146.0/146.0 kB 2.9 MB/s eta 0:00:00
Requirement already satisfied: requests in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from ccimport>=0.4.0->spconv-cu117) (2.28.1)
Collecting lark>=1.0.0
  Downloading lark-1.1.4-py3-none-any.whl (107 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 107.8/107.8 kB 2.2 MB/s eta 0:00:00
Requirement already satisfied: portalocker>=2.3.2 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from pccm>=0.4.0->spconv-cu117) (2.6.0)
Requirement already satisfied: termcolor in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from fire->spconv-cu117) (2.0.1)
Requirement already satisfied: six in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from fire->spconv-cu117) (1.16.0)
Requirement already satisfied: idna<4,>=2.5 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from requests->ccimport>=0.4.0->spconv-cu117) (3.4)
Requirement already satisfied: certifi>=2017.4.17 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from requests->ccimport>=0.4.0->spconv-cu117) (2022.9.24)
Requirement already satisfied: charset-normalizer<3,>=2 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from requests->ccimport>=0.4.0->spconv-cu117) (2.1.1)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from requests->ccimport>=0.4.0->spconv-cu117) (1.26.12)
Installing collected packages: ninja, lark, pybind11, fire, ccimport, pccm, cumm-cu117, spconv-cu117
Successfully installed ccimport-0.4.2 cumm-cu117-0.3.7 fire-0.4.0 lark-1.1.4 ninja-1.11.1 pccm-0.4.4 pybind11-2.10.1 spconv-cu117-2.2.6
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 

```

#
<br>

```bash
(env_point_net) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep spconv
(env_point_net) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep torch
torch               1.10.1
torchaudio          0.10.1
torchvision         0.11.2
(env_point_net) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep cuda
(env_point_net) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep cumm
```
#
<br>
- Move to diff CONDA Env -- env2_det2

```bash
(env_point_net) dhankar@dhankar-1:~/.../OpenPCDet$ conda activate env2_det2
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep cumm
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep torch
torch                       1.12.1
torchaudio                  0.10.1
torchvision                 0.13.1
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep cuda
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ pip list | grep spconv
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ python
Python 3.9.13 (main, Oct 13 2022, 21:15:33) 
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
[2]+  Stopped                 python
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 

```



#
<br>

> PRIMARY PAPER - “Pointrcnn: 3d object proposal generation
and detection from point cloud,”

- https://arxiv.org/abs/1907.03670
- https://github.com/sshaoshuai/PointRCNN


> S. Shi, X. Wang, and H. Li, “Pointrcnn: 3d object proposal generation
and detection from point cloud,”
- Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 770–779.
- https://ieeexplore.ieee.org/document/8954080
> PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud
#
> Abstract:
In this paper, we propose PointRCNN for 3D object detection from raw point cloud. The whole framework is composed of two stages: stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in the canonical coordinates to obtain the final detection results. Instead of generating proposals from RGB image or projecting point cloud to bird's view or voxels as previous methods do, our stage-1 sub-network directly generates a small number of high-quality 3D proposals from point cloud in a bottom-up manner via segmenting the point cloud of the whole scene into foreground points and background. The stage-2 sub-network transforms the pooled points of each proposal to canonical coordinates to learn better local spatial features, which is combined with global semantic features of each point learned in stage-1 for accurate box refinement and confidence prediction. Extensive experiments on the 3D detection benchmark of KITTI dataset show that our proposed architecture outperforms state-of-the-art methods with remarkable margins by using only point cloud as input. The code is available at https://github.com/sshaoshuai/PointRCNN.

#
<br>

> PRIMARY PAPER CITATIONS -- 
- https://ieeexplore.ieee.org/document/8954080/citations?tabFilter=papers#citations




#
<br>


- python -m torch.distributed.launch 
- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md

> then collectively synchronizing gradients using the AllReduce primitive. In HPC terminology, this model of execution is called Single Program Multiple Data or SPMD since the same application runs on all application but each one operates on different portions of the training dataset.

#
<br>

- https://discuss.pytorch.org/t/the-right-way-to-distribute-the-training-over-multiple-gpus-and-nodes/29277


#
<br>

#### FASTER -- RCNN -->> https://arxiv.org/pdf/2208.04171.pdf
DCNN architectures can be categorized into two groups:
two-stage detectors and one-stage detectors. Two-stage detec-
tors have a proposal detection stage where a set of bounding
box candidates are generated and a verification stage where
these bounding boxes are separately evaluated whether they
contain an object of a specific class. Examples of these
networks are R-CNN [54], SPPNet [55], Fast R-CNN [56],
Faster R-CNN [6].

#### One STAGE DETECTORS -->> https://arxiv.org/pdf/2208.04171.pdf
On the other hand, in the case of one-stage detectors, a
single neural network is applied to the full image that predicts
the bounding boxes straight away. The slow detection time
which is the biggest disadvantage of the two-stage detectors,
can be overcome with the one-stage approach. Detection time
is crucial for many applications especially but not exclusively
in the field of robotics or self-driving cars. 


#### DATA GENERATION Framework -->> https://arxiv.org/pdf/2208.04171.pdf
For data generation, the PyBullet [25] physics simulator was
used since it has an easy-to-use, intuitive API, including an
image renderer tool, and an integrated physics simulator where
the gravitational force can be easily simulated. It is important
to mention that one of the biggest advantages of domain
randomization over domain adaptation is that it is generally
faster as images do not need to be photo-realistic. In our case,
we could achieve less than 0.5s per image (generation) on a
GeForce RTX 2080 Ti GPU. With 4000 images it is around
33 minutes. In general, the time of dataset generation is an
important aspect of the method as in the industry, on many
occasions, it is not feasible to wait long hours or even days
to start the training.


#### SUMMARY OF RELATED WORKS -->> https://arxiv.org/pdf/2208.04171.pdf
Work Problem Input Domain DR/DA Base Model Simulator Synthetic
Images
Real Im-
ages
Results
Tobin [14] Detection RGB Shapes DR VGG-16 MuJoCo [20] 5K-50K 0 1.5 cm
Tobin [21] Grasping Depth YCB [48] DR CNNs MuJoCo [20] 2K / obj 0 80% success
Borrego [22] Detection RGB Shapes DR Faster R-CNN Gazebo [23] 9K 0 82% mAP50
9K 121 83% mAP50
SSD 9K 0 64% mAP50
9K 121 62% mAP50
Pashevich [12] Detection Depth Cubes DR ResNet-18 PyBullet [25] 2K 0 1.09±0.73 cm
Cup placing Cups - 0 15/20
James [28] Pick-and-place RGB,
joints
Cube DR CNNs V-REP [49] 100K-1M 0 ≥41% success
Devo [29] Navigation 2xRGB - DR CNNs UE4 [30] 540K 0 46% success
Chen [31] Detection RGB Street DA Faster RCNN - 10K 3K(unlab.) 38.97 AP50
Sankarana-
rayanan [34]
Segmentation RGB Street DA GAN - 25K 5K(unlab.) 37.1 mIOU
Tremblay [37] Detection RGB Street DR Faster R-CNN UE4 [30] 100K 0 78.1 AP50
100
