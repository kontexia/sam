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
        Extension("src.sgm", ["src/sgm.py"])
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("src.sgm", ["src/sgm.c"])
    ]

setup(
    name='sam',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    packages=['src']
 )
