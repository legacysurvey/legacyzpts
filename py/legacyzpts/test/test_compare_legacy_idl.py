import os 
from glob import glob

from legacyzpts.qa.compare_idlzpts import ZptResiduals, StarResiduals
from legacyzpts.fetch import fetch_targz

DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'

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

def test_zpts_decam():
  fetch_targz(os.path.join(DOWNLOAD_DIR,
                           'idl_legacy_data.tar.gz'), 
              get_data_dir())

  leg_fns= glob(os.path.join(get_data_dir(),
                             'idl_legacy_data',
                             'todays_version/c4d*-zpt.fits'))
  idl_fns= glob(os.path.join(get_data_dir(),
                             'idl_legacy_data',
                             'zeropoint-c4d*.fits'))
  zpt= ZptResiduals(camera='decam',
                    savedir=get_output_dir('zpts'),
                    leg_list=leg_fns,
                    idl_list=idl_fns)
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
  #zpt.plot_residuals(doplot='diff') 

    
def test_stars_decam():
  fetch_targz(os.path.join(DOWNLOAD_DIR,
                           'idl_legacy_data.tar.gz'), 
              get_data_dir())

  leg_fns= glob(os.path.join(get_data_dir(),
                             'idl_legacy_data',
                             'c4d*-star.fits'))
  idl_fns= glob(os.path.join(get_data_dir(),
                             'idl_legacy_data',
                             'matches-c4d*.fits'))
  star= StarResiduals(camera='decam',
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
  star.convert_legacy()
  star.match(ra_key='ccd_ra',dec_key='ccd_dec')
  star.plot_residuals(doplot='diff') 

if __name__ == "__main__":
  test_zpts_decam()
  test_stars_decam()
