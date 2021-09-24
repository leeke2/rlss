from setuptools import setup, Extension

setup(name='rlss', version='1.0',  \
      ext_modules=[Extension('rlss', ['rlssmodule.c'])])