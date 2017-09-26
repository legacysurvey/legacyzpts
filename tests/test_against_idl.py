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

def load_zpts(camera=None):
  assert(camera in CAMERAS)
  fetch_targz(os.path.join(DOWNLOAD_DIR,
                           'idl_legacy_data.tar.gz'), 
              get_data_dir())

  #leg_fns= glob(os.path.join(get_data_dir(),
  #                           'idl_legacy_data',
  #                           'todays_version/%s*-zpt.fits' % 
  #                              FN_SUFFIX[camera]))
  leg_fns= glob(os.path.join(os.path.dirname(__file__),
                             'testoutput','ccds',
                             'small_c4d_*oki*-zpt.fits'))
                             
  idl_fns= glob(os.path.join(get_data_dir(),
                             'idl_legacy_data',
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

    
def load_stars(camera=None):
  assert(camera in CAMERAS)
  fetch_targz(os.path.join(DOWNLOAD_DIR,
                           'idl_legacy_data.tar.gz'), 
              get_data_dir())

  #leg_fns= glob(os.path.join(get_data_dir(),
  #                           'idl_legacy_data',
  #                           '%s*-star.fits' %
  #                             FN_SUFFIX[camera]))
  leg_fns= glob(os.path.join(os.path.dirname(__file__),
                             'testoutput','ccds',
                             'small_c4d_*oki*-star.fits'))
  
  idl_fns= glob(os.path.join(get_data_dir(),
                             'idl_legacy_data',
                             'matches-%s*.fits' %
                               FN_SUFFIX[camera]))
  star= StarResiduals(camera=camera,
                      savedir=get_output_dir('stars'),
                      leg_list=leg_fns,
                      idl_list=idl_fns)
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

def test_zpts_decam_vs_idl():
  """Sanity check how close legacy values are to idl

  Args:
    zpts: ZptResidual object as returned by test_load_zpts
  """
  print("zpts: decam_vs_idl".upper())
  # Load, match
  zpts= load_zpts(camera='decam')

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

def test_stars_decam_vs_idl():
  """Sanity check how close legacy values are to idl

  Args:
    star: StarResidual object as returned by test_load_star
  """
  print("stars: decam_vs_idl".upper())
  # Load, match
  stars= load_stars(camera='decam')
  
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


if __name__ == "__main__":
  # Zpts
  #zpts= load_zpts(camera='decam', plot=False)
  test_zpts_decam_vs_idl()
  # Stars
  #stars= load_stars(camera='decam', plot=False)
  test_stars_decam_vs_idl()
