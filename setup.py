#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
#
# Standard imports
#
import glob
import os
import sys
from setuptools import setup, find_packages
#
# Begin setup
#
setup_keywords = dict()
#
# THESE SETTINGS NEED TO BE CHANGED FOR EVERY PRODUCT.
#
setup_keywords['name'] = 'legacyzpts'
setup_keywords['description'] = 'astrometric and photometric calibration'
setup_keywords['author'] = 'K. Burleigh, J. Moustakas'
setup_keywords['author_email'] = 'kburleigh@lbl.gov'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/legacysurvey/legacyzpts'
#
# END OF SETTINGS THAT NEED TO BE CHANGED.
#
setup_keywords['version'] = '0.1.0'
#
# Use README.rst as long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.rst'):
    with open('README.rst') as readme:
        setup_keywords['long_description'] = readme.read()
#
# Set other keywords for the setup function.  These are automated, & should
# be left alone unless you are an expert.
#
# Treat everything in bin/ except *.rst as a script to be installed.
#
if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
        if not os.path.basename(fname).endswith('.rst')]
setup_keywords['provides'] = [setup_keywords['name']]
#setup_keywords['requires'] = ['Python (>2.7.0)']
setup_keywords['install_requires'] = [
	'Python (>2.7.0)',
	'astropy',
	'Cython',
	'healpy',
	'h5py',
	'ipython',
	'jupyter',
	'matplotlib',
	'pandas',
	'psycopg2',
	'six',
	'Sphinx']
setup_keywords['dependency_links'] = [
	'https://github.com/legacysurvey/legacypipe.git@dr5.0#egg=legacypipe',
	'https://github.com/dstndstn/tractor/archive/dr5.2.tar.gz',
	'https://github.com/dstndstn/astrometry.net/releases/download/0.72/astrometry.net-0.72.tar.gz',
	]
#setup_keywords=['setup_requires']= ['setup.cfg','pytest-runner']
#setup_keywords['setup_cfg']=True
#setup_keywords['tests_require']= ['pytest']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages('py')
setup_keywords['package_dir'] = {'':'py'}
#
# setup_keywords['entry_points'] = {'console_scripts':['desiInstall = desiutil.install.main:main']}
#
# Run setup command.
#
setup(**setup_keywords)
