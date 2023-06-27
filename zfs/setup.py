# run with python setup.py build_ext

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# run with "python setup.py build_ext"
print(np.get_include())
ext_modules = cythonize([
    Extension("main", ["main.py"], include_dirs=[np.get_include()]),
    Extension("parallel", ["parallel.py"], include_dirs=[np.get_include()]),
    Extension("poscar", ["poscar.py"], include_dirs=[np.get_include()]),
    Extension("wavecar", ["wavecar.py"], include_dirs=[np.get_include()]),
    Extension("vasp", ["vasp.py"], include_dirs=[np.get_include()])
])
setup(py_modules=['numpy', 'scipy', 'mpi4py'],
      ext_modules=ext_modules)
