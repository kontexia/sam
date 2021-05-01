from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options
    from Cython.Build import cythonize

except ImportError:
    use_cython = False
else:
    use_cython = True
    Cython.Compiler.Options.annotate = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("src.sparse_distributed_representation", ["src/sparse_distributed_representation.py"]),
        Extension("src.neurons", ["src/neurons.py"])

    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("src.sparse_distributed_representation", ["src/sparse_distributed_representation.c"]),
        Extension("src.neurons", ["src/neurons.c"])
    ]

setup(
    name='sam',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    packages=['src']
 )
