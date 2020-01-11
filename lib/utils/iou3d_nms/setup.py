from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d_nms',
    ext_modules=[
        CUDAExtension('iou3d_nms_cuda', [
            'src/iou3d_nms.cpp',
            'src/iou3d_nms_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
