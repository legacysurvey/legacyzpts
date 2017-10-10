"""
Tests that reproduce survey-ccds tables from DR3, DR4

Compares -legacyzpts.fits tables to DR3, DR4 survey-ccds tables
Use DR3,DR4 tables because these were made with IDL zeropoints so this is a test of 
the input tables given to tractor that help make DR3,DR4
"""


import os 
from glob import glob
import numpy as np
from scipy import stats 
import fitsio
try:
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.libkd.spherematch import match_radec
except ImportError:
    pass

from legacyzpts.qa.compare_idlzpts import ZptResiduals, StarResiduals
from legacyzpts.fetch import fetch_targz
from legacyzpts.common import merge_tables_fns
from legacyzpts.legacy_zeropoints import cols_for_legacypipe_table

from tests.test_against_common import get_tolerance,PlotDifference,differenceChecker 


DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'
CAMERAS= ['decam','mosaic','90prime']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "90prime":"ksb"}

def get_data_dir():
  return os.path.join(os.path.dirname(__file__),
                      'testdata')

class LoadData(object):
  def __init__(self):
    """Download needed data"""
    fetch_targz(os.path.join(DOWNLOAD_DIR,
                             'surveyccds.tar.gz'), 
                get_data_dir())

  def legacypipe_matched_surveyccds(self,camera='decam',indir='ps1_gaia',
                                    prod=False):
    """returns matched -legacypipe table to DR3/4 surveyccds table for that camera

    Args:
      indir: the testoutput directory to read from
      prod: tests written to testoutput/ dir, if True it will look for production run
        outputs which are assumed to be copied to prodoutput/ dir

    Returns:
      leg,ccds: the -legacypipe, survey-ccds tables
    """
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])
    # -legacypipe table
    testoutput= 'testoutput'
    if prod:
      testoutput= testoutput.replace('test','prod')
    leg_dir= os.path.join(os.path.dirname(__file__),
                          testoutput,camera,
                          indir)
    if camera in ['decam','mosaic']:
      # Different idl zeropoints and surveyccds images
      leg_dir= os.path.join(leg_dir,'against_surveyccds')
    patt= os.path.join(leg_dir,
                       '*%s*-legacypipe.fits' % 
                          FN_SUFFIX[camera])
    print('pattern=%s' % patt)
    leg_fns= glob(patt)
    assert(len(leg_fns) > 0)
    leg= merge_tables_fns(leg_fns,textfile=False)
    # survey-ccds table
    if camera == 'decam':
      ccds_fn= os.path.join(get_data_dir(),'surveyccds',
                            'dr3','survey-ccds-decals.fits.gz')
    elif camera == 'mosaic':
      ccds_fn= os.path.join(get_data_dir(),'surveyccds',
                            'dr4','survey-ccds-mzls.fits.gz')
    elif camera == '90prime':
      ccds_fn= os.path.join(get_data_dir(),'surveyccds',
                            'dr4','survey-ccds-bass.fits.gz')
    ccds= fits_table(ccds_fn)
    # Match, big list first
    m1, m2, d12 = match_radec(ccds.ra, ccds.dec,
                              leg.ra, leg.dec,
                              1./3600.0,nearest=True)
    print('%d Matches' % len(m1))
    ccds.cut(m1)
    leg.cut(m2)
    return leg,ccds

#############
# TEST FUNCS
############

def test_legacypipe_table(camera='decam',indir='ps1_gaia', 
                          plot=False, prod=False):
    """checks that difference between legacypipe and surveyccds sufficiently small
    Args:
      camera: CAMERAS
      indir: the testoutput directory to read from
      prod: tests written to testoutput/ dir, if True it will look for production run
        outputs which are assumed to be copied to prodoutput/ dir
    """
    print("TESTING LEGACYPIPE %s" % camera)
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])
    # Matched tables
    leg,ccds= LoadData().legacypipe_matched_surveyccds(camera=camera,
                                                       indir=indir, prod=prod)
    # Check differences
    cols= cols_for_legacypipe_table(which='numeric')
    not_in_surveyccds= ['skyrms']
    if camera == '90prime':
      not_in_surveyccds+= ['expnum']
    for col in not_in_surveyccds:
      cols.remove(col)
    differenceChecker(data=leg, ref=ccds,
                      cols=cols, camera=camera,
                      legacyzpts_product='legacypipe')   
    # Plot
    if plot:
      cols= cols_for_legacypipe_table(which='nonzero_diff')
      not_in_surveyccds= ['skyrms']
      for col in not_in_surveyccds:
        print('removing col=%s' % col)
        cols.remove(col)
      PlotDifference(legacyzpts_product='legacypipe',
                     camera=camera,indir=indir,prod=prod,
                     against='surveyccds',
                     x=ccds, y=leg, cols=cols, 
                     xname='Surveyccds',yname='Legacy')
    assert(True)

def test_main():
  plot=False
  production=False
  test_legacypipe_table(camera='decam',indir='ps1_gaia',
                        plot=plot,prod=production)
  test_legacypipe_table(camera='mosaic',indir='ps1_gaia',
                        plot=plot, prod=production)
  test_legacypipe_table(camera='90prime',indir='ps1_gaia',
                        plot=plot,prod=production)
 

if __name__ == "__main__":
  test_main()
 
