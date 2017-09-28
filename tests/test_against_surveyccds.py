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



DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'
CAMERAS= ['decam','mosaic','bok']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "bok":"bs4"}

def get_data_dir():
  return os.path.join(os.path.dirname(__file__),
                      'testdata')

class LoadData(object):
  def __init__(self):
    """Download needed data"""
    fetch_targz(os.path.join(DOWNLOAD_DIR,
                             'surveyccds.tar.gz'), 
                get_data_dir())

  def legacypipe_matched_surveyccds(self,camera='decam',indir='ps1_gaia'):
    """returns matched -legacypipe table to DR3/4 surveyccds table for that camera

    Args:
      indir: the testoutput directory to read from

    Returns:
      leg,ccds: the -legacypipe, survey-ccds tables
    """
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])
    # -legacypipe table
    leg_dir= os.path.join(os.path.dirname(__file__),
                          'testoutput',camera,
                          indir,'against_surveyccds')
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

def test_legacypipe_table(camera='decam',indir='ps1_gaia'):
    """
    Args:
      indir: the testoutput directory to read from
    """
    print("TESTING LEGACYPIPE again survey-ccds")
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])
    # Matched tables
    leg,ccds= LoadData().legacypipe_matched_surveyccds(camera=camera,
                                                       indir=indir)
    # Tolerances
    numeric_cols= cols_for_legacypipe_table(which='numeric')
    not_in_surveyccds= ['skyrms']
    hard_to_compare= ['ccdraoff','ccddecoff'] + ['width','height']
    # Defalut is 0.01 unless specify here
    tol= {'fwhm':0.1,
          'ccdnmatch':0.5}
    #should_be_different= ['ccdnmatch']
    for col in numeric_cols:
      assert(col in leg.get_columns())
      assert(np.all(np.isfinite(leg.get(col))))
      if col in not_in_surveyccds + hard_to_compare:
        continue
      abs_rel_diff= np.abs(  (leg.get(col) - ccds.get(col))/ccds.get(col)  )
      print('col=',col,'leg=',leg.get(col),'ccds=',ccds.get(col))
      print('\ttol=%g, abs_rel_diff=' % tol.get(col,0.01),abs_rel_diff)
      assert(np.all( abs_rel_diff < tol.get(col,0.01) ))
    assert(True)



if __name__ == "__main__":
  #test_legacypipe_table(camera='decam',indir='ps1_gaia')
  test_legacypipe_table(camera='mosaic',indir='ps1_gaia')
  #test_legacypipe_table(camera='decam',indir='ps1_only')
  #test_legacypipe_table(camera='decam',indir='ps1_only')
