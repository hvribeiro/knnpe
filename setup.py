from setuptools import setup, find_packages

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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
