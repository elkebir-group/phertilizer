#! /usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='phertilizer',
    version='0.1.0',
    author='Leah Weber and Chuanyi Zhang',
    author_email='leahlw2@illinois.edu',
    description="Growing a clonal tree from ultra-low coverage single-cell DNA sequencing data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/elkebir-group/phertilizer',
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': ['phertilizer=phertilizer.run:main_cli'],
    },
    install_requires=[
        "numpy",
        "pandas",
        "numba", 
        "scipy",
        "networkx",
        "scikit-learn",
        "pygraphviz",
        "umap-learn"
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    license='MIT',
)