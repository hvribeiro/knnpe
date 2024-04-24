from setuptools import setup, Extension, find_packages

from distutils.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

class CTypesExtension(Extension):
    pass


class build_ext(_build_ext):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


class bdist_wheel_abi_none(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        return "py3", "none", plat


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="knnpe",
    version="0.1.2",
    author="L. G. J. M. Voltarelli, A. A. B. Pessa, L. Zunino, R. S. Zola, E. K. Lenzi, M. Perc, H. V. Ribeiro",
    author_email="hvr@dfi.uem.br",
    description="A Python package for characterizing unstructured data with the nearest neighbor permutation entropy",
    long_description=long_description,
    long_description_content_type="text/x-rst; charset=UTF-8",
    packages=find_packages(),
    ext_modules=[
        CTypesExtension(
            "knnpe/rwalk.rwalk",
            ["knnpe/rwalk/rwalk.c"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-lgsl', '-lgslcblas', '-lm','-lgomp']
        ),
    ],
    cmdclass={"build_ext": build_ext, "bdist_wheel": bdist_wheel_abi_none},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
