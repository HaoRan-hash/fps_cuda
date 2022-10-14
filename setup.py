from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='fps_cuda',
    ext_modules=[CUDAExtension('fps_cuda', 
                               ['fps.cu'])],
    cmdclass={'build_ext': BuildExtension}
)