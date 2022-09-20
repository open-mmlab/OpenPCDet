from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='slice_and_batch',
      ext_modules=[cpp_extension.CppExtension('slice_and_batch', ['slice_and_batch.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
