#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
from glob import glob
import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

class MyInstall(install):
    def run(self):
        print('MyInstall.run: calling "make -k"')
        subprocess.call(['make', '-k'])
        print('MyInstall.run: calling "make -k py"')
        subprocess.call(['make', '-k', 'py'])

        for cmd in ['make -k pyinstall',
                    'make -k install']:
	        dirnm = self.install_base
	        if dirnm is not None:
	            cmd += ' INSTALL_DIR="%s"' % dirnm
	        pybase = self.install_platlib
	        if pybase is not None:
	            pybase = os.path.join(pybase, 'astrometry')
	            cmd += ' PY_BASE_INSTALL_DIR="%s"' % pybase
	        py = sys.executable
	        if py is not None:
	            cmd += ' PYTHON="%s"' % py
	        print('Running:', cmd)
	        subprocess.call(cmd, shell=True)
	        install.run(self)

#
#
long_description = ''
if os.path.exists('README.rst'):
    with open('README.rst') as readme:
        long_description = readme.read()
#
# Run setup command.
#
setup(name= 'legacyzpts',
	  description='astrometric and photometric calibration',
	  long_description= long_description,
	  author='K. Burleigh, J. Moustakas',
	  author_email= 'kburleigh@lbl.gov',
	  license= 'BSD',
	  url= 'https://github.com/legacysurvey/legacyzpts',
	  version='0.1.0',
	  packages= find_packages('py'),
	  package_dir= {'':'py'})
	  #cmdclass={'install': MyInstall},)
