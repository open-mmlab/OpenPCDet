from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='slice_and_batch',
    ext_modules=[
        CUDAExtension('slice_and_batch_cuda', [
            'slice_and_batch.cpp',
            'slice_and_batch_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
