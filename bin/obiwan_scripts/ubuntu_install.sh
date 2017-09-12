#!/bin/bash

#sudo apt-get update
#sudo apt-get install git emacs python-setuptools python-dev build-essential
#sudo easy_install pip
#sudo pip install --upgrade virtualenv
## python3: python3 -m venv <myenvname>
#virtualenv $HOME/env_galsim --python=python2.7
source $HOME/env_galsim/bin/activate
export python_exe=`which python`
echo python_exe=$python_exe
# deactivate

sudo apt-get install -y graphviz 
sudo apt-get install -y texlive-latex-extra
sudo apt-get install -y dvipng
sudo apt-get install -y libnetpbm10
sudo apt-get install -y libnetpbm10-dev
sudo apt-get install -y netpbm
sudo apt-get install -y wcslib-dev
sudo apt-get install -y libz-dev
sudo apt-get install -y libcfitsio3
sudo apt-get install -y libcfitsio3-dev
sudo apt-get install -y swig
sudo apt-get install -y gsl-bin
sudo apt-get install -y libgsl0-dev
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y gfortran
sudo apt-get install -y liblapack-dev
sudo apt-get install -y libfftw3-dev
sudo apt-get install -y curl
sudo apt-get install -y python-tk
sudo apt-get install -y libcairo2-dev libpng12-dev libjpeg-dev zlib1g-dev
sudo apt-get install -y libbz2-dev

export WCSLIB_INC="-I/usr/include/wcslib"
export WCSLIB_LIB="-lwcs"

pip install Sphinx
pip install numpy
pip install scipy
pip install matplotlib 
pip install cython
pip install numpydoc
pip install astropy
pip install --no-deps git+https://github.com/astropy/photutils.git
pip install h5py
pip install pandas
pip install psycopg2
pip install scikit-image
pip install scikit-learn
pip install nose 
pip install future 
pip install pyyaml
#pip install coveralls
pip install -U pytest
pip install pytest-cov coveralls pytest-pep8
pip install ipython
pip install jupyter
pip install --no-deps --upgrade git+https://github.com/esheldon/fitsio.git#egg=fitsio
#- pip install -v --no-deps --upgrade git+https://github.com/dstndstn/tractor.git

export MYINSTALL=$HOME/myinstall
export MYREPO=$HOME/myrepo
mkdir -p $MYINSTALL $MYREPO

if [ ! -d "$MYREPO/astrometry.net" ]; then (cd $MYREPO && curl -SL https://github.com/dstndstn/astrometry.net/releases/download/0.72/astrometry.net-0.72.tar.gz -o astrometry.net.tar.gz && tar -xzf astrometry.net.tar.gz && mv astrometry.net-0.72 astrometry.net); fi
cd $MYREPO/astrometry.net && make install INSTALL_DIR=$MYINSTALL/astrometry
export PYTHONPATH=$MYINSTALL/astrometry/lib/python:$PYTHONPATH
export PATH=$MYINSTALL/astrometry/lib/python/astrometry/util:$PATH
export PATH=$MYINSTALL/astrometry/lib/python/astrometry/blind:$PATH
cd $HOME && python -c "import astrometry; print(astrometry.__file__)"

if [ ! -d "$MYREPO/tractor" ]; then (cd $MYREPO && curl -SL https://github.com/dstndstn/tractor/archive/dr5.2.tar.gz -o tractor.tar.gz && tar -xzf tractor.tar.gz && mv tractor-dr5.2 tractor); fi
cd $MYREPO/tractor && make && python setup.py install --prefix=$MYINSTALL/tractor
export PYTHONPATH=$MYINSTALL/tractor/lib/python2.7/site-packages:$PYTHONPATH
cd $HOME && python -c "import tractor; print(tractor.__file__)"

mkdir -p $MYREPO/dust/maps
cd $MYREPO/dust/maps && wget -c http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_ngp.fits
cd $MYREPO/dust/maps && wget -c http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_sgp.fits
export DUST_DIR=$MYREPO/dust

echo DONE


