#!/usr/bin/env python

import os
from setuptools import setup, Command
from shutil import rmtree

base_path = os.path.dirname(os.path.abspath(__file__))

long_description = """
Integer data types lack special values for -inf, inf and NaN. Especially
NaN as an indication for missing data would be useful in many scientific contexts.

Of course there is numpy.ma.MaskedArray around for the very same reason. Nevertheless,
it might sometimes be annoying to carry a separate mask array around. And in those cases,
using a set of numpy-compatible functions for the same job will do just fine.

This package provides such an implementation for several standard numpy functions, that 
treat integer arrays in such a way, that the lowest negative integer resembles NaN.
"""


class Clean(Command):
    description = "clean up temporary files from 'build' command"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for folder in ('build', 'dist', 'intnan.egg-info'):
            path = os.path.join(base_path, folder)
            if os.path.isdir(path):
                print("removing '{}' (and everything under it)".format(path))
                if not self.dry_run:
                    rmtree(path)
        self._rm_walk()

    def _rm_walk(self):
        for path, dirs, files in os.walk(base_path):
            if any(p.startswith('.') for p in path.split(os.path.sep)):
                # Skip hidden directories like the git folder right away
                continue
            if path.endswith('__pycache__'):
                print("removing '{}' (and everything under it)".format(path))
                if not self.dry_run:
                    rmtree(path)
            else:
                for fname in files:
                    if fname.endswith('.pyc') or fname.endswith('.so'):
                        fpath = os.path.join(path, fname)
                        print("removing '{}'".format(fpath))
                        if not self.dry_run:
                            os.remove(fpath)


setup(name='intnan',
      version="0.1",
      author="Michael Loeffler",
      author_email="ml@occam.com.ua",
      license='BSD',
      description="Function collection for handling integers with NaNs",
      long_description=long_description,
      long_description_content_type="text/x-rst",
      url="https://github.com/ml31415/intnan",
      download_url="https://github.com/ml31415/intnan/archive/master.zip",
      keywords=[ "integer", "nan", "missing values", "intnan"],
      packages=["intnan"],
      install_requires=[],
      setup_requires=["pytest-runner"],
      tests_require=["pytest", "numpy", "numba"],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   ],
      cmdclass={"clean": Clean},
      project_urls={"Source": "https://github.com/ml31415/intnan"}
)
