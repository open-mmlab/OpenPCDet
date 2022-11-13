



##### DEMO 

#
<br>

- got the KITTI data from here -->
- https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d


> python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/data/pv_rcnn_8369.pth \
    --data_path /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/000009.bin

#
<br>


##### DEMO -1 

#
<br>


```bash
(env2_det2) dhankar@dhankar-1:~/.../tools$ 
(env2_det2) dhankar@dhankar-1:~/.../tools$ python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
>     --ckpt /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/data/pv_rcnn_8369.pth \
>     --data_path /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/000009.bin

2022-11-14 00:00:35,770   INFO  -----------------Quick Demo of OpenPCDet-------------------------
2022-11-14 00:00:35,770   INFO  Total number of samples: 	1
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1639180487213/work/aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-11-14 00:01:15,667   INFO  ==> Loading parameters from checkpoint /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/data/pv_rcnn_8369.pth to CPU
2022-11-14 00:01:16,370   INFO  ==> Done (loaded 367/367)
2022-11-14 00:01:16,649   INFO  Visualized sample index: 	1
Traceback (most recent call last):
  File "/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/tools/demo.py", line 112, in <module>
    main()
  File "/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/tools/demo.py", line 100, in main
    V.draw_scenes(
  File "/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/tools/visual_utils/visualize_utils.py", line 154, in draw_scenes
    fig = visualize_pts(points)
  File "/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/tools/visual_utils/visualize_utils.py", line 77, in visualize_pts
    fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/mayavi/tools/figure.py", line 64, in figure
    engine = get_engine()
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/mayavi/tools/engine_manager.py", line 94, in get_engine
    return self.new_engine()
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/mayavi/tools/engine_manager.py", line 139, in new_engine
    check_backend()
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/mayavi/tools/engine_manager.py", line 42, in check_backend
    raise ImportError(msg)
ImportError: Could not import backend for traitsui.  Make sure you
        have a suitable UI toolkit like PyQt/PySide or wxPython
        installed.
(env2_det2) dhankar@dhankar-1:~/.../tools$ 

```

# 000009.png - /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/image_2/
# 000009.bin - /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/
# 000009.txt - /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/calib/

cp /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/testing/velodyne/000009.bin /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/

cp /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/testing/velodyne/000094.bin /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/

cp /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/testing/velodyne/000039.bin /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/

cp /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/testing/velodyne/000011.bin /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/

cp /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/testing/velodyne/000028.bin /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/

cd /home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data/KITTI/object/testing/velodyne/















(env2_det2) dhankar@dhankar-1:~/.../data$ 
(env2_det2) dhankar@dhankar-1:~/.../data$ tree -d
.
├── KITTI
│   ├── ImageSets
│   └── object
│       ├── 11
│       │   ├── testing
│       │   │   └── image_2
│       │   └── training
│       │       └── image_2
│       ├── testing
│       │   ├── calib
│       │   ├── image_2
│       │   └── velodyne
│       └── training
│           ├── calib
│           ├── calib1
│           │   └── training
│           ├── image_2
│           ├── label_2
│           └── velodyne
├── testing
│   └── velodyne
└── training
    └── velodyne

23 directories
(env2_det2) dhankar@dhankar-1:~/.../data$ pwd
/home/dhankar/temp/11_22/original_PointRCNN/PointRCNN/data
(env2_det2) dhankar@dhankar-1:~/.../data$ 


#
<br>

> python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/data/pv_rcnn_8369.pth \
    --data_path ${POINT_CLOUD_DATA}

#
<br>


```bash
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/data
(base) dhankar@dhankar-1:~/.../data$ ls -lahtr
total 69M
drwxrwxr-x  3 dhankar dhankar 4.0K Nov  6 15:11 waymo
drwxrwxr-x  3 dhankar dhankar 4.0K Nov  6 15:11 lyft
drwxrwxr-x  3 dhankar dhankar 4.0K Nov  6 15:11 kitti
drwxrwxr-x 11 dhankar dhankar 4.0K Nov 13 20:31 ..
-rw-rw-r--  1 dhankar dhankar  19M Nov 13 21:14 pointpillar_7728.pth
-rw-rw-r--  1 dhankar dhankar  51M Nov 13 21:24 pv_rcnn_8369.pth
drwxrwxr-x  5 dhankar dhankar 4.0K Nov 13 21:25 .
(base) dhankar@dhankar-1:~/.../data$ 
```


#
<br>



##### Initial Install Terminal Prints - Terminal Log File for the Original OpenPCDet
- This is posted here within the FORK repo so that dont malign the code of the Original Repo 
- this here FORK is being used to Document own Process / Experiments 
- Search below with -- [1/13] , [2/13] ... etc etc ...
- Terminal Prints for -- MAYAVI_Install

#
<br>


###### Terminal Print STARTs for -- Init_Install_Setup.py_develop

> (env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ python setup.py develop


#
<br>

```bash
(base) dhankar@dhankar-1:~/.../OpenPCDet$ conda activate env_det2
(env_det2) dhankar@dhankar-1:~/.../OpenPCDet$ conda activate env2_det2
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ python setup.py develop
running develop
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/setuptools/command/easy_install.py:156: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
running egg_info
creating pcdet.egg-info
writing pcdet.egg-info/PKG-INFO
writing dependency_links to pcdet.egg-info/dependency_links.txt
writing requirements to pcdet.egg-info/requires.txt
writing top-level names to pcdet.egg-info/top_level.txt
writing manifest file 'pcdet.egg-info/SOURCES.txt'
reading manifest file 'pcdet.egg-info/SOURCES.txt'
adding license file 'LICENSE'
writing manifest file 'pcdet.egg-info/SOURCES.txt'
running build_ext
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/utils/cpp_extension.py:782: UserWarning: The detected CUDA version (11.0) has a minor version mismatch with the version that was used to compile PyTorch (11.3). Most likely this shouldn't be a problem.
  warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
building 'pcdet.ops.iou3d_nms.iou3d_nms_cuda' extension
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src
Emitting ninja build file /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/4] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=iou3d_nms_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[2/4] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_cpu.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_cpu.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=iou3d_nms_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp: In function ‘int boxes_iou_bev_cpu(at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp:242:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *boxes_a = boxes_a_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp:243:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *boxes_b = boxes_b_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp:244:49: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *ans_iou = ans_iou_tensor.data<float>();
                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[3/4] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=iou3d_nms_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp: In function ‘int boxes_overlap_bev_gpu(at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:54:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes_a);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:55:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes_b);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:56:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(ans_overlap);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:61:54: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes_a_data = boxes_a.data<float>();
                                                      ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:62:54: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes_b_data = boxes_b.data<float>();
                                                      ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:63:56: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float * ans_overlap_data = ans_overlap.data<float>();
                                                        ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp: In function ‘int boxes_iou_bev_gpu(at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:74:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes_a);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:75:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes_b);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:76:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(ans_iou);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:81:54: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes_a_data = boxes_a.data<float>();
                                                      ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:82:54: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes_b_data = boxes_b.data<float>();
                                                      ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:83:48: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float * ans_iou_data = ans_iou.data<float>();
                                                ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp: In function ‘int nms_gpu(at::Tensor, at::Tensor, float)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:93:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:97:50: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes_data = boxes.data<float>();
                                                  ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:98:40: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     long * keep_data = keep.data<long>();
                                        ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp: In function ‘int nms_normal_gpu(at::Tensor, at::Tensor, float)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:143:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:147:50: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes_data = boxes.data<float>();
                                                  ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:148:40: warning: ‘T* at::Tensor::data() const [with T = long int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     long * keep_data = keep.data<long>();
                                        ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp:7:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[4/4] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms_api.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/iou3d_nms/src/iou3d_nms_api.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms_api.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=iou3d_nms_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
creating build/lib.linux-x86_64-3.9
creating build/lib.linux-x86_64-3.9/pcdet
creating build/lib.linux-x86_64-3.9/pcdet/ops
creating build/lib.linux-x86_64-3.9/pcdet/ops/iou3d_nms
g++ -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -shared -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_cpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms_api.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.o -L/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/lib -L/usr/local/cuda-11.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp -o build/lib.linux-x86_64-3.9/pcdet/ops/iou3d_nms/iou3d_nms_cuda.cpython-39-x86_64-linux-gnu.so
building 'pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda' extension
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/src
Emitting ninja build file /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=roiaware_pool3d_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[2/2] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=roiaware_pool3d_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp: In function ‘int roiaware_pool3d_gpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:55:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *rois_data = rois.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:56:45: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *pts_data = pts.data<float>();
                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:57:61: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *pts_feature_data = pts_feature.data<float>();
                                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:58:41: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *argmax_data = argmax.data<int>();
                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:59:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *pts_idx_of_voxels_data = pts_idx_of_voxels.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:60:63: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *pooled_features_data = pooled_features.data<float>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp: In function ‘int roiaware_pool3d_gpu_backward(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:87:69: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *pts_idx_of_voxels_data = pts_idx_of_voxels.data<int>();
                                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:88:47: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *argmax_data = argmax.data<int>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:89:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_out_data = grad_out.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:90:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_in_data = grad_in.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp: In function ‘int points_in_boxes_gpu(at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:111:51: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *boxes = boxes_tensor.data<float>();
                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:112:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *pts = pts_tensor.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:113:65: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *box_idx_of_points = box_idx_of_points_tensor.data<int>();
                                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp: In function ‘int points_in_boxes_cpu(at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:155:51: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *boxes = boxes_tensor.data<float>();
                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:156:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *pts = pts_tensor.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:157:53: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *pts_indices = pts_indices_tensor.data<int>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp:9:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
creating build/lib.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d
g++ -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -shared -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.o -L/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/lib -L/usr/local/cuda-11.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp -o build/lib.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/roiaware_pool3d_cuda.cpython-39-x86_64-linux-gnu.so
building 'pcdet.ops.roipoint_pool3d.roipoint_pool3d_cuda' extension
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/src
Emitting ninja build file /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=roipoint_pool3d_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[2/2] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=roipoint_pool3d_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp: In function ‘int roipool3d_gpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:5:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:16:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:29:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:5:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:16:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:30:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(boxes3d);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:5:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:16:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:31:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(pts_feature);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:5:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:16:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:32:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(pooled_features);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:5:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:16:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:33:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(pooled_empty_flag);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:42:46: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * xyz_data = xyz.data<float>();
                                              ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:43:54: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * boxes3d_data = boxes3d.data<float>();
                                                      ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:44:62: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float * pts_feature_data = pts_feature.data<float>();
                                                              ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:45:64: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float * pooled_features_data = pooled_features.data<float>();
                                                                ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:46:64: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int * pooled_empty_flag_data = pooled_empty_flag.data<int>();
                                                                ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
creating build/lib.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d
g++ -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -shared -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/src/roipoint_pool3d_kernel.o -L/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/lib -L/usr/local/cuda-11.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp -o build/lib.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/roipoint_pool3d_cuda.cpython-39-x86_64-linux-gnu.so
building 'pcdet.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda' extension
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src
Emitting ninja build file /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp: In function ‘int ball_query_wrapper_stack(int, int, float, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:32:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:33:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:34:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:35:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:37:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz = new_xyz_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:38:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *xyz = xyz_tensor.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:39:71: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
                                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:40:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:41:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[2/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9

                from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[2/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp: In function ‘void three_nn_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:42:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(unknown_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:43:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(unknown_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:44:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(known_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:45:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(known_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:46:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(dist2_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:47:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:52:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *unknown = unknown_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:53:71: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *unknown_batch_cnt = unknown_batch_cnt_tensor.data<int>();
                                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:54:51: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *known = known_tensor.data<float>();
                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:55:67: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *known_batch_cnt = known_batch_cnt_tensor.data<int>();
                                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:56:45: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *dist2 = dist2_tensor.data<float>();
                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:57:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp: In function ‘void three_interpolate_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:70:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:71:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:72:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(weight_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:73:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(out_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:77:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *features = features_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:78:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *weight = weight_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:79:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:80:41: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *out = out_tensor.data<float>();
                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp: In function ‘void three_interpolate_grad_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:93:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grad_out_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:94:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:95:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(weight_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:96:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grad_features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:100:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_out = grad_out_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:101:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *weight = weight_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:102:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:103:61: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_features = grad_features_tensor.data<float>();
                                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[3/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp: In function ‘int group_points_grad_wrapper_stack(int, int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:33:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grad_out_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:34:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:35:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:36:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(features_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:37:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grad_features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:39:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_out = grad_out_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:40:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:41:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:42:73: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
                                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:43:61: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_features = grad_features_tensor.data<float>();
                                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp: In function ‘int group_points_wrapper_stack(int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:54:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:55:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(features_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:56:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:57:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:15:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:58:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(out_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:60:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *features = features_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:61:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:62:73: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
                                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:63:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:64:41: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *out = out_tensor.data<float>();
                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[4/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp: In function ‘int query_stacked_local_neighbor_idxs_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, float, int, int)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:47:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(support_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:48:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:49:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:50:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:51:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(stack_neighbor_idxs_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,

                        ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:51:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(stack_neighbor_idxs_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:52:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(start_len_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:53:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(cumsum_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:55:63: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *support_xyz = support_xyz_tensor.data<float>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:56:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:57:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz = new_xyz_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:58:71: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
                                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:59:69: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *stack_neighbor_idxs = stack_neighbor_idxs_tensor.data<int>();
                                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:60:49: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *start_len = start_len_tensor.data<int>();
                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:61:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *cumsum = cumsum_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp: In function ‘int query_three_nn_by_stacked_local_idxs_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:88:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(support_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:89:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:90:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_grid_centers_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:91:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_grid_idxs_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:92:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_grid_dist2_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:93:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(stack_neighbor_idxs_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:94:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(start_len_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:96:63: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *support_xyz = support_xyz_tensor.data<float>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:97:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz = new_xyz_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:98:81: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz_grid_centers = new_xyz_grid_centers_tensor.data<float>();
                                                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:99:65: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *new_xyz_grid_idxs = new_xyz_grid_idxs_tensor.data<int>();
                                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:100:71: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *new_xyz_grid_dist2 = new_xyz_grid_dist2_tensor.data<float>();
                                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:101:69: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *stack_neighbor_idxs = stack_neighbor_idxs_tensor.data<int>();
                                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:102:49: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *start_len = start_len_tensor.data<int>();
                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp: In function ‘int vector_pool_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int, int, float, int, int, int, int, int)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:132:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(support_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:133:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(support_features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:134:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:135:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:136:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:137:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:138:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_local_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:139:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(point_cnt_of_grid_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:140:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grouped_idxs_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:142:63: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *support_xyz = support_xyz_tensor.data<float>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:143:73: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *support_features = support_features_tensor.data<float>();
                                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:144:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:145:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz = new_xyz_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:146:71: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
                                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:147:59: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *new_features = new_features_tensor.data<float>();
                                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:148:61: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *new_local_xyz = new_local_xyz_tensor.data<float>();
                                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:149:65: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *point_cnt_of_grid = point_cnt_of_grid_tensor.data<int>();
                                                                 ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:150:55: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *grouped_idxs = grouped_idxs_tensor.data<int>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp: In function ‘int vector_pool_grad_wrapper_stack(at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:178:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grad_new_features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:179:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(point_cnt_of_grid_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:180:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grouped_idxs_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:18:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:29:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:181:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(grad_support_features_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:190:75: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_new_features = grad_new_features_tensor.data<float>();
                                                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:191:71: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *point_cnt_of_grid = point_cnt_of_grid_tensor.data<int>();
                                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:192:61: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *grouped_idxs = grouped_idxs_tensor.data<int>();
                                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:193:77: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_support_features = grad_support_features_tensor.data<float>();
                                                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp:11:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[5/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp: In function ‘int farthest_point_sampling_wrapper(int, int, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:24:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(points_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:25:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(temp_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:26:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:28:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *points = points_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:29:43: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *temp = temp_tensor.data<float>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:30:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp: In function ‘int stack_farthest_point_sampling_wrapper(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:41:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(points_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:42:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(temp_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:43:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(idx_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:44:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_batch_cnt_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:7:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:18:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:45:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(num_sampled_points_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:49:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *points = points_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:50:43: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *temp = temp_tensor.data<float>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:51:57: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:52:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:53:67: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *num_sampled_points = num_sampled_points_tensor.data<int>();
                                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[6/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp: In function ‘int voxel_query_wrapper_stack(int, int, int, int, int, float, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:11:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:22:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:28:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_coords_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:11:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:22:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:29:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(point_indices_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:11:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:22:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:30:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:11:15: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
   if (!x.type().is_cuda()) { \
               ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:22:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:31:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:33:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz = new_xyz_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:34:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *xyz = xyz_tensor.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:35:57: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *new_coords = new_coords_tensor.data<int>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:36:63: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *point_indices = point_indices_tensor.data<int>();
                                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:37:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.cpp:1:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[7/13] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/pointnet2_api.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/pointnet2_api.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/pointnet2_api.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
[8/13] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[9/13] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[10/13] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/group_points_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/group_points_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[11/13] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/sampling_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/sampling_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[12/13] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[13/13] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_stack_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
creating build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2
creating build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack
g++ -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -shared -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/group_points.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/group_points_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/interpolate_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/pointnet2_api.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/sampling.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/sampling_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query_gpu.o -L/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/lib -L/usr/local/cuda-11.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp -o build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_stack_cuda.cpython-39-x86_64-linux-gnu.so
building 'pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda' extension
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch
creating /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src
Emitting ninja build file /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/build.ninja...
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/9] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp: In function ‘int group_points_grad_wrapper_fast(int, int, int, int, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:18:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_points = grad_points_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:19:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:20:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_out = grad_out_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp: In function ‘int group_points_wrapper_fast(int, int, int, int, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:30:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *points = points_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:31:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:32:41: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *out = out_tensor.data<float>();
                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[2/9] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp: In function ‘int ball_query_wrapper_fast(int, int, int, float, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:15:16: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
    if (!x.type().is_cuda()) { \
                ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:31:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(new_xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:15:16: warning: ‘at::DeprecatedTypeProperties& at::Tensor::type() const’ is deprecated: Tensor.type() is deprecated. Instead use Tensor.options(), which in many cases (e.g. in a constructor) is a drop-in replacement. If you were using data from type(), that is now available from Tensor itself, so instead of tensor.type().scalar_type(), use tensor.scalar_type() instead and instead of tensor.type().backend() use tensor.device(). [-Wdeprecated-declarations]
    if (!x.type().is_cuda()) { \
                ^
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:26:24: note: in expansion of macro ‘CHECK_CUDA’
 #define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
                        ^~~~~~~~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:32:5: note: in expansion of macro ‘CHECK_INPUT’
     CHECK_INPUT(xyz_tensor);
     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:194:30: note: declared here
   DeprecatedTypeProperties & type() const {
                              ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:33:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *new_xyz = new_xyz_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:34:47: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *xyz = xyz_tensor.data<float>();
                                               ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:35:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[3/9] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp: In function ‘void three_nn_wrapper_fast(int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:20:55: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *unknown = unknown_tensor.data<float>();
                                                       ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:21:51: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *known = known_tensor.data<float>();
                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:22:45: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *dist2 = dist2_tensor.data<float>();
                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/


                                                   ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:22:45: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *dist2 = dist2_tensor.data<float>();
                                             ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:23:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp: In function ‘void three_interpolate_wrapper_fast(int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:35:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *points = points_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:36:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *weight = weight_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:37:41: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *out = out_tensor.data<float>();
                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:38:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp: In function ‘void three_interpolate_grad_wrapper_fast(int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:50:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_out = grad_out_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:51:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *weight = weight_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:52:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_points = grad_points_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:53:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[4/9] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp: In function ‘int gather_points_wrapper_fast(int, int, int, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:16:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *points = points_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:17:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:18:41: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *out = out_tensor.data<float>();
                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp: In function ‘int gather_points_grad_wrapper_fast(int, int, int, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:28:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *grad_out = grad_out_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:29:43: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const int *idx = idx_tensor.data<int>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:30:57: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *grad_points = grad_points_tensor.data<float>();
                                                         ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp: In function ‘int farthest_point_sampling_wrapper(int, int, int, at::Tensor, at::Tensor, at::Tensor)’:
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:40:53: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     const float *points = points_tensor.data<float>();
                                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:41:43: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     float *temp = temp_tensor.data<float>();
                                           ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
/home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:42:37: warning: ‘T* at::Tensor::data() const [with T = int]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     int *idx = idx_tensor.data<int>();
                                     ^
In file included from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Tensor.h:3:0,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/Context.h:4,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/ATen.h:9,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/types.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/input-archive.h:6,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/archive.h:3,
                 from /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/serialize/tensor.h:3,
                 from /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp:8:
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/ATen/core/TensorBody.h:216:7: note: declared here
   T * data() const {
       ^~~~
[5/9] c++ -MMD -MF /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/pointnet2_api.o.d -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -I/home/dhankar/anaconda3/envs/env2_det2/include -fPIC -O2 -isystem /home/dhankar/anaconda3/envs/env2_det2/include -fPIC -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/pointnet2_api.cpp -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/pointnet2_api.o -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14
[6/9] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[7/9] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[8/9] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
[9/9] /usr/local/cuda-11.0/bin/nvcc  -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/TH -I/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/include/THC -I/usr/local/cuda-11.0/include -I/home/dhankar/anaconda3/envs/env2_det2/include/python3.9 -c -c /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/src/group_points_gpu.cu -o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/group_points_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=pointnet2_batch_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 -std=c++14
creating build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch
g++ -pthread -B /home/dhankar/anaconda3/envs/env2_det2/compiler_compat -shared -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath,/home/dhankar/anaconda3/envs/env2_det2/lib -Wl,-rpath-link,/home/dhankar/anaconda3/envs/env2_det2/lib -L/home/dhankar/anaconda3/envs/env2_det2/lib /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/ball_query_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/group_points.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/group_points_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/interpolate_gpu.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/pointnet2_api.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/sampling.o /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet/build/temp.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu.o -L/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/lib -L/usr/local/cuda-11.0/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp -o build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/pointnet2_batch_cuda.cpython-39-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-3.9/pcdet/ops/iou3d_nms/iou3d_nms_cuda.cpython-39-x86_64-linux-gnu.so -> pcdet/ops/iou3d_nms
copying build/lib.linux-x86_64-3.9/pcdet/ops/roiaware_pool3d/roiaware_pool3d_cuda.cpython-39-x86_64-linux-gnu.so -> pcdet/ops/roiaware_pool3d
copying build/lib.linux-x86_64-3.9/pcdet/ops/roipoint_pool3d/roipoint_pool3d_cuda.cpython-39-x86_64-linux-gnu.so -> pcdet/ops/roipoint_pool3d
copying build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_stack_cuda.cpython-39-x86_64-linux-gnu.so -> pcdet/ops/pointnet2/pointnet2_stack
copying build/lib.linux-x86_64-3.9/pcdet/ops/pointnet2/pointnet2_batch/pointnet2_batch_cuda.cpython-39-x86_64-linux-gnu.so -> pcdet/ops/pointnet2/pointnet2_batch
Creating /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/pcdet.egg-link (link to .)
Adding pcdet 0.6.0+f221374 to easy-install.pth file

Installed /home/dhankar/temp/11_22/original_open_pc_det/OpenPCDet
Processing dependencies for pcdet==0.6.0+f221374
Searching for SharedArray
Reading https://pypi.org/simple/SharedArray/
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning:  is an invalid version and will not be supported in a future release
  warnings.warn(
Downloading https://files.pythonhosted.org/packages/9f/d8/19760d3bfbcae4d96b4ced9710673a91dd49a56eff1479e480fb5e31e642/SharedArray-3.2.2.tar.gz#sha256=eb1b1ae3953864f0b5ceb30e27cd832d19525ece71176c246cc5d30a82f0f8ed
Best match: SharedArray 3.2.2
Processing SharedArray-3.2.2.tar.gz
Writing /tmp/easy_install-o6ckr1cw/SharedArray-3.2.2/setup.cfg
Running SharedArray-3.2.2/setup.py -q bdist_egg --dist-dir /tmp/easy_install-o6ckr1cw/SharedArray-3.2.2/egg-dist-tmp-ykx2s_3t
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
zip_safe flag not set; analyzing archive contents...
__pycache__.SharedArray.cpython-39: module references __file__
creating /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/SharedArray-3.2.2-py3.9-linux-x86_64.egg
Extracting SharedArray-3.2.2-py3.9-linux-x86_64.egg to /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Adding SharedArray 3.2.2 to easy-install.pth file

Installed /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/SharedArray-3.2.2-py3.9-linux-x86_64.egg
Searching for easydict
Reading https://pypi.org/simple/easydict/
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning:  is an invalid version and will not be supported in a future release
  warnings.warn(
Downloading https://files.pythonhosted.org/packages/55/83/0d1ee7962f3ba3fbe9eebe67eb484f6745995c9af045c0ebe5f33564cba0/easydict-1.10.tar.gz#sha256=11dcb2c20aaabbfee4c188b4bc143ef6be044b34dbf0ce5a593242c2695a080f
Best match: easydict 1.10
Processing easydict-1.10.tar.gz
Writing /tmp/easy_install-2dadc9v3/easydict-1.10/setup.cfg
Running easydict-1.10/setup.py -q bdist_egg --dist-dir /tmp/easy_install-2dadc9v3/easydict-1.10/egg-dist-tmp-xp8j1xla
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
creating /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/easydict-1.10-py3.9.egg
Extracting easydict-1.10-py3.9.egg to /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Adding easydict 1.10 to easy-install.pth file

Installed /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/easydict-1.10-py3.9.egg
Searching for tensorboardX
Reading https://pypi.org/simple/tensorboardX/
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/pkg_resources/__init__.py:116: PkgResourcesDeprecationWarning:  is an invalid version and will not be supported in a future release
  warnings.warn(
Downloading https://files.pythonhosted.org/packages/96/47/9004f6b182920e921b6937a345019c9317fda4cbfcbeeb2af618b3b7a53e/tensorboardX-2.5.1-py2.py3-none-any.whl#sha256=8808133ccca673cd04076f6f2a85cf2d39bb2d0393a0f20d0f9cbb06d472b57e
Best match: tensorboardX 2.5.1
Processing tensorboardX-2.5.1-py2.py3-none-any.whl
Installing tensorboardX-2.5.1-py2.py3-none-any.whl to /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Adding tensorboardX 2.5.1 to easy-install.pth file

Installed /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/tensorboardX-2.5.1-py3.9.egg
Searching for numba
Reading https://pypi.org/simple/numba/
Downloading https://files.pythonhosted.org/packages/60/14/5dbefc1cf3b6a4c36968e7391c341b32226c5d00757efd61fe5f3d96a32e/numba-0.56.4-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=0240f9026b015e336069329839208ebd70ec34ae5bfbf402e4fcc8e06197528e
Best match: numba 0.56.4
Processing numba-0.56.4-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
Installing numba-0.56.4-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl to /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Adding numba 0.56.4 to easy-install.pth file
Installing pycc script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing numba script to /home/dhankar/anaconda3/envs/env2_det2/bin

Installed /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/numba-0.56.4-py3.9-linux-x86_64.egg
Searching for llvmlite
Reading https://pypi.org/simple/llvmlite/
Downloading https://files.pythonhosted.org/packages/94/ae/a24ff97a39a8b4c60b93c63ccde867249c1d5dab03d790e85d64f99c0db3/llvmlite-0.39.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=60f8dd1e76f47b3dbdee4b38d9189f3e020d22a173c00f930b52131001d801f9
Best match: llvmlite 0.39.1
Processing llvmlite-0.39.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
Installing llvmlite-0.39.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl to /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Adding llvmlite 0.39.1 to easy-install.pth file

Installed /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/llvmlite-0.39.1-py3.9-linux-x86_64.egg
Searching for tqdm==4.64.1
Best match: tqdm 4.64.1
Adding tqdm 4.64.1 to easy-install.pth file
Installing tqdm script to /home/dhankar/anaconda3/envs/env2_det2/bin

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for scikit-image==0.19.3
Best match: scikit-image 0.19.3
Adding scikit-image 0.19.3 to easy-install.pth file
Installing skivi script to /home/dhankar/anaconda3/envs/env2_det2/bin

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for PyYAML==6.0
Best match: PyYAML 6.0
Adding PyYAML 6.0 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for numpy==1.23.4
Best match: numpy 1.23.4
Adding numpy 1.23.4 to easy-install.pth file
Installing f2py script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing f2py3 script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing f2py3.9 script to /home/dhankar/anaconda3/envs/env2_det2/bin

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for imageio==2.22.2
Best match: imageio 2.22.2
Adding imageio 2.22.2 to easy-install.pth file
Installing imageio_download_bin script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing imageio_remove_bin script to /home/dhankar/anaconda3/envs/env2_det2/bin

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for scipy==1.9.3
Best match: scipy 1.9.3
Adding scipy 1.9.3 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for networkx==2.8.7
Best match: networkx 2.8.7
Adding networkx 2.8.7 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for tifffile==2022.10.10
Best match: tifffile 2022.10.10
Adding tifffile 2022.10.10 to easy-install.pth file
Installing lsm2bin script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing tiff2fsspec script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing tiffcomment script to /home/dhankar/anaconda3/envs/env2_det2/bin
Installing tifffile script to /home/dhankar/anaconda3/envs/env2_det2/bin

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for Pillow==9.2.0
Best match: Pillow 9.2.0
Adding Pillow 9.2.0 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for packaging==21.3
Best match: packaging 21.3
Adding packaging 21.3 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for PyWavelets==1.4.1
Best match: PyWavelets 1.4.1
Adding PyWavelets 1.4.1 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for protobuf==3.19.6
Best match: protobuf 3.19.6
Adding protobuf 3.19.6 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for setuptools==59.5.0
Best match: setuptools 59.5.0
Adding setuptools 59.5.0 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Searching for pyparsing==3.0.9
Best match: pyparsing 3.0.9
Adding pyparsing 3.0.9 to easy-install.pth file

Using /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages
Finished processing dependencies for pcdet==0.6.0+f221374
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 
```
###### Terminal Print Ends for -- Init_Install_Setup.py_develop

> (env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ python setup.py develop



###### Terminal Prints STARTS for -- MAYAVI_Install

- pip install mayavi

#
<br>

#
```bash

(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 
(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ pip install mayavi
Collecting mayavi
  Downloading mayavi-4.8.1.tar.gz (20.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.6/20.6 MB 12.8 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: packaging in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from mayavi) (21.3)
Collecting envisage
  Downloading envisage-6.1.0-py3-none-any.whl (280 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 280.8/280.8 kB 3.9 MB/s eta 0:00:00
Collecting traits>=6.0.0
  Downloading traits-6.4.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 16.1 MB/s eta 0:00:00
Collecting traitsui>=7.0.0
  Downloading traitsui-7.4.2-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 14.6 MB/s eta 0:00:00
Requirement already satisfied: pygments in /home/dhankar/.local/lib/python3.9/site-packages (from mayavi) (2.10.0)
Collecting pyface>=6.1.1
  Downloading pyface-7.4.2-py3-none-any.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 12.2 MB/s eta 0:00:00
Collecting vtk
  Using cached vtk-9.2.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (79.3 MB)
Collecting apptools
  Downloading apptools-5.2.0-py3-none-any.whl (229 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.2/229.2 kB 2.8 MB/s eta 0:00:00
Requirement already satisfied: numpy in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from mayavi) (1.23.4)
Collecting configobj
  Using cached configobj-5.0.6.tar.gz (33 kB)
  Preparing metadata (setup.py) ... done
Requirement already satisfied: setuptools in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from envisage->mayavi) (59.5.0)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from packaging->mayavi) (3.0.9)
Collecting wslink>=1.0.4
  Using cached wslink-1.9.1-py3-none-any.whl (28 kB)
Requirement already satisfied: matplotlib>=2.0.0 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from vtk->mayavi) (3.6.1)
Requirement already satisfied: python-dateutil>=2.7 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from matplotlib>=2.0.0->vtk->mayavi) (2.8.2)
Requirement already satisfied: contourpy>=1.0.1 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from matplotlib>=2.0.0->vtk->mayavi) (1.0.5)
Requirement already satisfied: pillow>=6.2.0 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from matplotlib>=2.0.0->vtk->mayavi) (9.2.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from matplotlib>=2.0.0->vtk->mayavi) (1.4.4)
Requirement already satisfied: fonttools>=4.22.0 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from matplotlib>=2.0.0->vtk->mayavi) (4.38.0)
Requirement already satisfied: cycler>=0.10 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from matplotlib>=2.0.0->vtk->mayavi) (0.11.0)
Collecting aiohttp<4
  Using cached aiohttp-3.8.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
Requirement already satisfied: six in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from configobj->apptools->mayavi) (1.16.0)
Collecting aiosignal>=1.1.2
  Using cached aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
Collecting attrs>=17.3.0
  Using cached attrs-22.1.0-py2.py3-none-any.whl (58 kB)
Collecting yarl<2.0,>=1.0
  Using cached yarl-1.8.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (264 kB)
Collecting frozenlist>=1.1.1
  Using cached frozenlist-1.3.3-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (158 kB)
Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from aiohttp<4->wslink>=1.0.4->vtk->mayavi) (2.1.1)
Collecting multidict<7.0,>=4.5
  Using cached multidict-6.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (114 kB)
Collecting async-timeout<5.0,>=4.0.0a3
  Using cached async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
Requirement already satisfied: idna>=2.0 in /home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages (from yarl<2.0,>=1.0->aiohttp<4->wslink>=1.0.4->vtk->mayavi) (3.4)
Building wheels for collected packages: mayavi, configobj
  Building wheel for mayavi (pyproject.toml) ... done
  Created wheel for mayavi: filename=mayavi-4.8.1-cp39-cp39-linux_x86_64.whl size=16135190 sha256=898daed6fdf1aace901c1c1dd14bf1d04836a319d96a49db88b9731106cd08c7
  Stored in directory: /home/dhankar/.cache/pip/wheels/5e/9c/2b/fa9f58121e0d2c709417504c8ebe52b07e4d859537fa52eda9
  Building wheel for configobj (setup.py) ... done
  Created wheel for configobj: filename=configobj-5.0.6-py3-none-any.whl size=34547 sha256=6c6d9180c3e4631442f455dc4622058d231f10c685428f4fecae2df2661392d7
  Stored in directory: /home/dhankar/.cache/pip/wheels/4b/35/53/dfa4d3a4196794cb0a777a97c68dcf02b073d33de9c135d72a

Successfully built mayavi configobj

Installing collected packages: traits, multidict, frozenlist, configobj, attrs, async-timeout, yarl, pyface, aiosignal, traitsui, aiohttp, wslink, apptools, vtk, envisage, mayavi

Successfully installed aiohttp-3.8.3 aiosignal-1.3.1 apptools-5.2.0 async-timeout-4.0.2 attrs-22.1.0 configobj-5.0.6 envisage-6.1.0 frozenlist-1.3.3 mayavi-4.8.1 multidict-6.0.2 pyface-7.4.2 traits-6.4.1 traitsui-7.4.2 vtk-9.2.2 wslink-1.9.1 yarl-1.8.1

(env2_det2) dhankar@dhankar-1:~/.../OpenPCDet$ 
```

###### Terminal Prints ENDS for -- MAYAVI_Install

