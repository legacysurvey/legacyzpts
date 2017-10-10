"""
Tests that reproduce IDL zeropoints and matches tables

Converts -zpt.fits tables to columns and units of IDL zeropoints
then compare to acutal idl zeropoints
"""

import os 
from glob import glob
import numpy as np
from scipy import stats 

from legacyzpts.qa.compare_idlzpts import ZptResiduals, StarResiduals
from legacyzpts.fetch import fetch_targz
from legacyzpts.legacy_zeropoints import cols_for_converted_zpt_table,cols_for_converted_star_table

from tests.test_against_common import PlotDifference,differenceChecker


DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'
CAMERAS= ['decam','mosaic','90prime']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "90prime":"ksb"}

def get_output_dir(case=None):
  """return path to directory for outputs

  Args:
    case: zpts,stars,shared
      shared is special in that any test case may write to it
      to pass info between test cases
  """
  assert(case in ['zpts','stars','shared'])
  dr= os.path.join(os.path.dirname(__file__),
                   'testoutput_%s' % case)
  if not os.path.exists(dr):
    os.makedirs(dr)
  return dr

def get_data_dir():
  return os.path.join(os.path.dirname(__file__),
                      'testdata')

class LoadData(object):
  def __init__(self):
    """Download needed data"""
    fetch_targz(os.path.join(DOWNLOAD_DIR,
                             'good_idlzpts_data.tar.gz'), 
                get_data_dir())
    fetch_targz(os.path.join(DOWNLOAD_DIR,
                             'good_legacyzpts_data.tar.gz'), 
                get_data_dir())

  def zpts_old_but_good(self,camera=None):
    assert(camera in CAMERAS)

    leg_fns= glob(os.path.join(get_data_dir(),
                               'good_legacyzpts_data',
                               'small_%s*-zpt.fits' % 
                                  FN_SUFFIX[camera]))
                               
    idl_fns= glob(os.path.join(get_data_dir(),
                               'good_idlzpts_data',
                               'zeropoint-%s*.fits' %
                                  FN_SUFFIX[camera]))
    zpt= ZptResiduals(camera=camera,
                      savedir=get_output_dir('zpts'),
                      leg_list=leg_fns,
                      idl_list=idl_fns,
                      loadable=False)
    zpt.load_data()
    # Writes dict mapping zpt table  quanitiesthat has info needed when
    zpt.write_json_expnum2var('exptime',
                 os.path.join(get_output_dir('shared'),
                              'expnum2exptime.json'))
    zpt.write_json_expnum2var('gain',
                 os.path.join(get_output_dir('shared'),
                              'expnum2gain.json'))

    zpt.convert_legacy()
    zpt.match(ra_key='ccdra',dec_key='ccddec')
    assert(len(zpt.legacy.data) == len(zpt.idl.data) )
    return zpt 

  def zpts_new(self,camera='decam',indir='ps1_gaia',
               prod=False):
    """
    Args:
      indir: the testoutput directory to read from
      prod: tests written to testoutput/ dir, if True it will look for production run
        outputs which are assumed to be copied to prodoutput/ dir
    """
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])

    testoutput= 'testoutput'
    if prod:
      testoutput= testoutput.replace('test','prod')
    leg_dir= os.path.join(os.path.dirname(__file__),
                          testoutput,camera,
                          indir,'against_idl')
    print('leg_dir=%s' % leg_dir)
    leg_fns= glob(os.path.join(leg_dir,
                               '*%s*-zpt.fits' % 
                                  FN_SUFFIX[camera]))
    if camera in ['decam','mosaic']:
      zeropoint_patt= 'zeropoint-%s*.fits' %\
                         FN_SUFFIX[camera]
    elif camera in ['90prime']:
      zeropoint_patt= 'bass-zpt-all-2016dec06.fits'
    idl_fns= glob(os.path.join(get_data_dir(),
                               'good_idlzpts_data',
                               zeropoint_patt))
    assert(len(leg_fns) > 0 and len(idl_fns) > 0)
    zpt= ZptResiduals(camera=camera,
                      savedir= leg_dir,
                      leg_list=leg_fns,
                      idl_list=idl_fns,
                      loadable=False)
    zpt.load_data()
    zpt.convert_legacy()
    zpt.match(ra_key='ccdra',dec_key='ccddec')
    assert(len(zpt.legacy.data) == len(zpt.idl.data) )
    return zpt 

  def stars_old_but_good(self,camera=None):
    assert(camera in CAMERAS)
    leg_fns= glob(os.path.join(get_data_dir(),
                               'good_legacyzpts_data',
                               'small_%s*-star.fits' %
                                 FN_SUFFIX[camera]))
    
    idl_fns= glob(os.path.join(get_data_dir(),
                               'good_idlzpts_data',
                               'matches-%s*.fits' %
                                 FN_SUFFIX[camera]))
    star= StarResiduals(camera=camera,
                        savedir=get_output_dir('stars'),
                        leg_list=leg_fns,
                        idl_list=idl_fns,
                        loadable=False)
    star.load_data()
    star.add_legacy_field('exptime', 
              os.path.join(get_output_dir('shared'),
                           'expnum2exptime.json'))
    star.add_legacy_field('gain', 
              os.path.join(get_output_dir('shared'),
                           'expnum2gain.json'))
    star.add_legacy_field('dmagall')
    star.convert_legacy()
    star.match(ra_key='ccd_ra',dec_key='ccd_dec')
    assert(len(star.legacy.data) == len(star.idl.data) )
    return star

  def stars_new(self,camera='decam',
                star_table='photom',indir='ps1_gaia'):
    """Two tables are possible, photom and astrom

    Args:
      star_table: photom or astrom
      indir: the testoutput directory to read from
    """
    assert(camera in CAMERAS)
    assert(star_table in ['photom','astrom'])
    assert(indir in ['ps1_gaia','ps1_only'])
    leg_dir= os.path.join(os.path.dirname(__file__),
                          'testoutput',camera,
                           indir,'against_idl')
    print('leg_dir=%s' % leg_dir)
    leg_fns= glob(os.path.join(leg_dir,
                               '*%s*-star-%s.fits' % (FN_SUFFIX[camera],star_table)))

    idl_fns= glob(os.path.join(get_data_dir(),
                               'good_idlzpts_data',
                               'matches-%s*.fits' %
                                 FN_SUFFIX[camera]))
    assert(len(leg_fns) > 0 and len(idl_fns) > 0)
    star= StarResiduals(camera=camera, star_table=star_table,
                        savedir= leg_dir,
                        leg_list=leg_fns,
                        idl_list=idl_fns,
                        loadable=False)
    star.load_data()
    star.convert_legacy()
    star.match(ra_key='ccd_ra',dec_key='ccd_dec')
    assert(len(star.legacy.data) == len(star.idl.data) )
    return star

class CheckDifference(object):
  """checks that legazpts tables are sufficienlty close to idl zpts tables
  
  Note: supports zpts and stars tables
  """

  def zeropoints(self,camera='decam',zpts=None):
    """Sanity check how close legacy values are to idl

    Args:
      camera: CAMERAS
      zpts: ZptResidual object as returned by LoadData().zpts_*()
    """
    print('see differenceChecker')
  
  def stars(self,camera='decam',stars=None):
    """Sanity check how close legacy values are to idl

    Args:
      camera: CAMERAS
      star: StarResidual object as returned by test_load_star
    """
    print("stars: decam_vs_idl".upper())
    
    # Tolerances
    tol= {'ccd_ra':1.e-5,
          'ccd_dec':1.e-5}
    ylim_dict= {key: (-tol[key],tol[key])
                for key in tol.keys()}
    stars.plot_residuals(doplot='diff',
                         use_keys=['ccd_ra','ccd_dec'],
                         ylim_dict=ylim_dict)
    # Test
    for col in ['ccd_ra','ccd_dec']:
      diff,_,_= stats.sigmaclip(stars.legacy.data.get(col) - 
                                stars.idl.data.get(col))
      print('require %s < %g, stats=' % (col,tol[col]), 
            stats.describe( np.abs(diff) ))
      assert(np.all( np.abs(diff) < tol[col]))


#############
# TEST FUNCS
############


def test_zpt_table(camera='decam',indir='ps1_gaia',
                   plot=False,prod=False):
  """Convert -zpt to idl names and units then compare to IDL zeropoint- table
  
  Args:
    camera:
    indir:
    prod: tests written to testoutput/ dir, if True it will look for production run
      outputs which are assumed to be copied to prodoutput/ dir
    plot: set to True to make plot of all quantities with 
      non-zero differences
  """
  print("TESTING ZPT")
  assert(camera in CAMERAS)
  assert(indir in ['ps1_gaia','ps1_only'])
  # Load and Match legacyzpts to IDLzpts
  zpts= LoadData().zpts_new(camera=camera,indir=indir,
                            prod=prod)
  #return zpts
  cols= cols_for_converted_zpt_table(which='numeric')
  ignore_cols= ['avsky','fwhm']
  if camera == '90prime':
    ignore_cols+= ['expnum']
  for col in ignore_cols:
    cols.remove(col)
  differenceChecker(data=zpts.legacy.data, ref=zpts.idl.data,
                    cols=cols, camera=camera,
                    legacyzpts_product='zpt')   
  if plot:
    cols= cols_for_converted_zpt_table(which='nonzero_diff')
    ignore_cols= ['avsky','fwhm']
    for col in ignore_cols:
      cols.remove(col)
    PlotDifference(legacyzpts_product='zpt',camera=camera,
                   indir=indir,prod=prod, against='idl',
                   x=zpts.idl.data, y=zpts.legacy.data, 
                   cols= cols,
                   xname='IDL',yname='Legacy')
  
  assert(True)

def test_star_table(camera='decam',indir='ps1_gaia',
                    star_table='photom',plot=False):
  """Convert -star to idl names and units then compare to IDL matches- table
  
  Args:
    camera:
    indir:
    star_table: there are two -star.fits tables: photom and astrom
    plot: set to True to make plot of all quantities with 
      non-zero differences
  """
  assert(camera in CAMERAS)
  assert(indir in ['ps1_gaia','ps1_only'])
  assert(star_table in ['photom','astrom'])
  print("TESTING STAR %s" % star_table)
  stars= LoadData().stars_new(camera=camera, indir=indir,
                              star_table=star_table)
  cols= cols_for_converted_star_table(star_table= star_table,
                                      which='numeric')
  skip_keys= ['nmatch','gmag']
  if star_table == 'astrom':
    skip_keys += ['ps1_'+band for band in ['g','r','i','z']]
  for key in skip_keys: 
    cols.remove(key)
  differenceChecker(data=stars.legacy.data, ref=stars.idl.data,
                    cols=cols, camera=camera,
                    legacyzpts_product='star-%s' % star_table)   
  #DecamEqualsIDL().stars(stars)
  if plot:
    cols= cols_for_converted_star_table(star_table= star_table,
                                        which='nonzero_diff')
    # Redundant or not in my file
    for key in set(cols).intersection(set(skip_keys)): 
      cols.remove(key)
    PlotDifference(legacyzpts_product='star-%s' % star_table,
                   camera=camera,indir=indir,against='idl',
                   x=stars.idl.data, y=stars.legacy.data, 
                   cols= cols,
                   xname='IDL',yname='Legacy')
  assert(True)


#def test_decam_zpts_old_but_good():
#  print("OLD BUT GOOD: decam zpts")
#  # Load and Match legacyzpts to IDLzpts
#  zpts= LoadData().zpts_old_but_good(camera='decam')
#  CheckTolerance().zeropoints(camera='decam',zpts=zpts)
#  assert(True)
#
#def test_decam_stars_old_but_good():
#  print("OLD BUT GOOD: decam stars")
#  # Load and Match legacy to IDL
#  stars= LoadData().stars_old_but_good(camera='decam')
#  CheckTolerance().stars(camera='decam',stars=stars)
#  assert(True)



if __name__ == "__main__":
  #test_decam_zpts_old_but_good()
  #test_decam_stars_old_but_good()
  
  
  # Default settings
  plot=False
  production=False
  test_zpt_table(camera='decam',indir='ps1_gaia',
                 plot=plot,prod=production)
  #for star_table in ['photom','astrom']:
  #  test_star_table(camera='decam',indir='ps1_gaia',
  #                  star_table=star_table,plot=plot)
  
  test_zpt_table(camera='mosaic',indir='ps1_gaia',
                 plot=plot,prod=production)
  #for star_table in ['photom','astrom']:
  #  test_star_table(camera='mosaic',indir='ps1_gaia',
  #                  star_table=star_table,plot=plot)
  
  test_zpt_table(camera='90prime',indir='ps1_gaia',
                 plot=plot,prod=production)
  
  
  #test_decam_stars_new(indir='ps1_gaia')
  # eBOSS DR5
  #test_zpt_table(camera='decam',indir='ps1_only')
  #test_decam_stars_new(indir='ps1_only')
  
  #test_zpt_table(camera='mosaic',indir='ps1_gaia')
  #test_mosaic_stars_new(indir='ps1_gaia')
  #test_decam_stars_new(indir='ps1_gaia')
  # eBOSS DR5
  #test_zpt_table(camera='decam',indir='ps1_only')
  #test_decam_stars_new(indir='ps1_only')
  
  #test_zpt_table(camera='mosaic',indir='ps1_gaia')
  #test_mosaic_stars_new(indir='ps1_gaia')
