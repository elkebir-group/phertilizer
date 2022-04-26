#! /usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='phertilizer',
    packages=["phertilizer"],
    description="growing a clonal tree from ultra-low coverage single-cell DNA sequencing data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.0',
    url='http://github.com/elkebir-group/phertilizer',
    author='Leah Weber and Chuanyi Zhang',
    author_email='leahlw2@illinois.edu',
    python_requires='>=3.7',
    entry_points = {
        'console_scripts': ['phertilizer= phertilizer.run:main_cli'],
    },
    install_requires=[
        "numpy",
        "pandas",
        "numba", 
        "scipy",
        "networkx",
        "scikit-learn",
        "pygraphviz",
    ],
)