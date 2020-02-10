#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='project3',
      version='0.1.0',
      license='GPL',
      description='Generic framework for historical document processing',
      install_requires=[
        'tensorflow-gpu',
        'opencv-python>=4.0.1',
        'numpy==1.16.2',
        'scikit-learn==0.20.3',
        'scikit-image==0.15.0',
        'pandas==0.24.2',
        'seaborn==0.9.0'
      ],
      zip_safe=False)
