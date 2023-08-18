#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E. N. Aslinger

from setuptools import setup, find_packages
import pathlib

must_test = []  # tests required
must_have = ['scanpy', 'pertpy', 'pandas', 'numpy']  # dependencies
pkg_was_here = pathlib.Path(__file__).parent.resolve()

with open('README.md') as info:
    readme = info.read()

setup(name='crispr',
      version='0.1.0',
      description='CRISPR workflow',
      url='http://github.com/easlinger/crispr',
      author='Elizabeth N. Aslinger',
      packages=find_packages(),
      install_requires=must_have,
      test_suite='tests',
      tests_require=['pytest'],
      long_description=readme,
      zip_safe=False,
      python_requires='>=3.6')
