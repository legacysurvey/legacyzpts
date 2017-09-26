import os 
from glob import glob
import numpy as np
from scipy import stats 

from legacyzpts.qa.compare_idlzpts import ZptResiduals, StarResiduals
from legacyzpts.fetch import fetch_targz


DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'
CAMERAS= ['decam','mosaic','bok']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "bok":"bs4"}

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

  def zpts_good_but_old(self,camera=None):
    assert(camera in CAMERAS)

    leg_fns= glob(os.path.join(get_data_dir(),
                               'good_legacyzpts_data',
                               '%s*-zpt.fits' % 
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

  def zpts_new(self,camera=None):
    assert(camera in CAMERAS)

    leg_dir= os.path.join(os.path.dirname(__file__),
                          'testoutput','new_legacyzpts_data')
    leg_fns= glob(os.path.join(leg_dir,
                               'small_%s*-zpt.fits' % 
                                  FN_SUFFIX[camera]))
                               
    idl_fns= glob(os.path.join(get_data_dir(),
                               'good_idlzpts_data',
                               'zeropoint-%s*.fits' %
                                  FN_SUFFIX[camera]))
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

 
  def stars_old_but_good(camera=None):
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

  def stars_new(camera=None,which='photom'):
    """Two tables are possible, photom and astrom
    """
    assert(camera in CAMERAS)
    assert(which in ['photom','astrom'])
    leg_dir= os.path.join(os.path.dirname(__file__),
                          'testoutput','new_legacyzpts_data')
    leg_fns= glob(os.path.join(leg_dir,
                               'small_c4d_*oki*-star-%s.fits' % which))
    
    idl_fns= glob(os.path.join(get_data_dir(),
                               'good_idlzpts_data',
                               'matches-%s*.fits' %
                                 FN_SUFFIX[camera]))
    star= StarResiduals(camera=camera,
                        savedir= leg_dir,
                        leg_list=leg_fns,
                        idl_list=idl_fns,
                        loadable=False)
    star.load_data()
    star.convert_legacy()
    star.match(ra_key='ccd_ra',dec_key='ccd_dec')
    assert(len(star.legacy.data) == len(star.idl.data) )
    return star



class DecamEqualsIDL(object):
  """checks that legazpts tables for decam are within tolerance compared to idl zpts tables
  
  Note: supports zpts and stars tables
  """

  def zeropoints(self,zpts):
    """Sanity check how close legacy values are to idl

    Args:
      zpts: ZptResidual object as returned by LoadData().zpts_*()
    """
    print("zpts: decam_vs_idl".upper())

    # Tolerances
    tol= {'ccdzpt':0.004,
          'ccdnmatch':20,
          'ccdskycounts':0.1}
    ylim_dict= {key: (-tol[key],tol[key])
                for key in tol.keys()}
    zpts.plot_residuals(doplot='diff',
                        use_keys=['ccdzpt','ccdnmatch',
                                  'ccdskycounts'],
                        ylim_dict=ylim_dict)
    # Test
    for col in ['ccdzpt']:
      diff,_,_= stats.sigmaclip(zpts.legacy.data.get(col) - 
                                zpts.idl.data.get(col))
      print('require %s < %g, stats=' % (col,tol[col]), 
            stats.describe( np.abs(diff) ))
      assert(np.all( np.abs(diff) < tol[col]))

    diff= np.abs(zpts.legacy.data.ccdnmatch -
                 zpts.idl.data.ccdnmatch)
    print('require %s < %g, stats=' % ('ccdnmatch',tol['ccdnmatch']), 
          stats.describe(diff ))
    assert(np.all( np.abs(zpts.legacy.data.ccdnmatch - 
                          zpts.idl.data.ccdnmatch) 
                          < tol['ccdnmatch']))
    
    filt= np.char.strip(zpts.legacy.data.filter)
    isGR= ((filt == 'g') |
           (filt == 'r'))
    isZ= (filt == 'z')
    diff= np.abs(zpts.legacy.data.ccdskycounts - 
                 zpts.idl.data.ccdskycounts)
    print('require gr %s < %g, stats=' % ('ccdskycounts', tol['ccdskycounts']/100.), 
          stats.describe( diff[isGR] ))
    assert(np.all( diff[isGR] < tol['ccdskycounts']/100.))
    
    print('require z %s < %g, stats=' % ('ccdskycounts', tol['ccdskycounts']), 
          stats.describe( diff[isZ] ))
    assert(np.all( diff[isZ] < tol['ccdskycounts']))

  def stars(self,stars):
    """Sanity check how close legacy values are to idl

    Args:
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

def test_decam_zpts_old_but_good():
  print("OLD BUT GOOD: decam zpts")
  # Load and Match legacyzpts to IDLzpts
  zpts= LoadData().zpts_old_but_good(camera='decam')
  DecamEqualsIDL().zeropoints(zpts)
  assert(True)

def test_decam_zpts_new():
  print("NEW: decam zpts")
  # Load and Match legacyzpts to IDLzpts
  zpts= LoadData().zpts_new(camera='decam')
  DecamEqualsIDL().zeropoints(zpts)
  assert(True)

def test_decam_stars_old_but_good():
  print("OLD BUT GOOD: decam stars")
  # Load and Match legacy to IDL
  stars= LoadData().stars_old_but_good(camera='decam')
  DecamEqualsIDL().stars(stars)
  assert(True)

def test_decam_stars_new():
  print("NEW: decam stars")
  # Load and Match legacy to IDL
  print('PHOTOM table')
  stars= LoadData().stars_new(camera='decam',which='photom')
  DecamEqualsIDL().stars(stars)
  print('ASRTROM table')
  stars= LoadData().stars_new(camera='decam',which='astrom')
  DecamEqualsIDL().stars(stars)
  assert(True)



if __name__ == "__main__":
  test_decam_zpts_old_but_good()
  test_decam_star_old_but_good()
  #test_decam_zpts_new()
  #test_decam_stars_new()
