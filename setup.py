from distutils.core import setup
from setuptools import find_packages

setup(name='acnets',
      version='2020.10.1',
      packages=find_packages('python'),
      package_dir={'': 'python'},
      long_description=open('README.md').read(),
      )
