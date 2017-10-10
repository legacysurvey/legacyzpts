import os
from glob import glob
import fitsio

from legacyzpts.legacy_zeropoints import main,get_parser
from legacyzpts.fetch import fetch_targz
from astrometry.util.fits import fits_table, merge_tables

DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'

CAMERAS= ['decam','mosaic','90prime']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "90prime":"bs4"}
PS1_GAIA_ARGS= ['--ps1_pattern','tests/testdata/chunks-qz-star-v3/ps1-%(hp)05d.fits',
                '--ps1_gaia_pattern','tests/testdata/chunks-ps1-gaia/chunk-%(hp)05d.fits']

def download_ccds():
    outdir= os.path.join(os.path.dirname(__file__),
                         'testdata')
    for targz in ['ccds_decam.tar.gz','ccds_mosaic.tar.gz',
                  'ccds_90prime.tar.gz',
                  'chunks-qz-star-v3.tar.gz','chunks-ps1-gaia.tar.gz']:
      fetch_targz(os.path.join(DOWNLOAD_DIR,
                               targz), 
                  outdir)

def run_and_check_outputs(image_list,cmd_line,outdir):
  """Runs legacyzpts and checks that expected files were written
  
  Args:
    image_list: list of small images to run on
    cmd_line: like ['--camera','decam','--outdir'], etc.
    outdir: where to write the zpts and stars tables
  """
  assert(len(image_list) > 0)
  parser= get_parser()
  args = parser.parse_args(args=cmd_line)
  main(image_list=image_list, args=args)
  # check output files exits
  for fn in image_list:
    base= os.path.basename(fn).replace(".fits.fz","")
    assert( os.path.exists(
                os.path.join(outdir,
                             base+"-debug-zpt.fits")))
    assert( os.path.exists(
                os.path.join(outdir,
                             base+"-debug-star-photom.fits")))
    assert( os.path.exists(
                os.path.join(outdir,
                             base+"-debug-star-astrom.fits")))
    assert( os.path.exists(
                os.path.join(outdir,
                             base+"-debug-legacypipe.fits")))


def test_decam(inSurveyccds=False, ps1_only=False):
    """Runs at least 1 CCD per band

    Args:
      inSurveyccds: True to run CCDs that are in the surveyccds file for DR3 or DR4
        False will be used to test again idl zeropoints instead of surveyccds
      ps1_only: True to use ps1 for astrometry and photometry
    """
    print('RUNNING LEGACYZPTS: default settings')
    download_ccds()
    if inSurveyccds:
      uniq_dir= 'against_surveyccds'
      img_patt= 'small_c4d_150409*ooi*.fits.fz'
    else:
      uniq_dir= 'against_idl'
      img_patt= 'small_c4d_17032*oki*.fits.fz'
    if ps1_only:
      ps1_gaia_dir= 'ps1_only'
      extra_args= ['--ps1_only']
    else:
      ps1_gaia_dir= 'ps1_gaia'
      extra_args= []
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','decam',
                          ps1_gaia_dir, uniq_dir)
    patt= os.path.join(os.path.dirname(__file__),
                       'testdata','ccds_decam',
                       img_patt)
    print('pattern=%s' % patt)
    fns= glob( patt)
    assert(len(fns) > 0)
    cmd_line=['--camera', 'decam','--outdir', outdir, 
              '--not_on_proj', '--debug'] + extra_args
    run_and_check_outputs(image_list=fns, cmd_line=cmd_line,
                          outdir=outdir)
    #run_and_check_outputs(image_list=[fns[0]], cmd_line=cmd_line,
    #                      outdir=outdir)

def test_mosaic(inSurveyccds=False, ps1_only=False):
    """Runs at least 1 CCD per band

    Args:
      inSurveyccds: True to run CCDs that are in the surveyccds file for DR3 or DR4
        False will be used to test again idl zeropoints instead of surveyccds
      ps1_only: True to use ps1 for astrometry and photometry
    """
    print('RUNNING LEGACYZPTS: default settings')
    download_ccds()
    if inSurveyccds:
      uniq_dir= 'against_surveyccds'
      img_patt= 'k4m*160605_042430*ooi*.fits.fz'
    else:
      uniq_dir= 'against_idl'
      img_patt= 'k4m*170221_072831*ooi*.fits.fz'
    if ps1_only:
      ps1_gaia_dir= 'ps1_only'
      extra_args= ['--ps1_only']
    else:
      ps1_gaia_dir= 'ps1_gaia'
      extra_args= []
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','mosaic',
                          ps1_gaia_dir,uniq_dir)
    fns= glob( os.path.join(os.path.dirname(__file__),
                            'testdata','ccds_mosaic',
                            img_patt))
    assert(len(fns) > 0)
    cmd_line=['--camera', 'mosaic','--outdir', outdir, 
              '--not_on_proj','--debug'] #+ PS1_GAIA_ARGS + extra_args
    run_and_check_outputs(image_list=fns, cmd_line=cmd_line,
                          outdir=outdir)
    #run_and_check_outputs(image_list=[fns[0]], cmd_line=cmd_line,
    #                      outdir=outdir)

def test_90prime(inSurveyccds=False, ps1_only=False):
    """Runs at least 1 CCD per band

    Args:
      inSurveyccds: True to run CCDs that are in the surveyccds file for DR3 or DR4
        False will be used to test again idl zeropoints instead of surveyccds
      ps1_only: True to use ps1 for astrometry and photometry
    """
    print('RUNNING LEGACYZPTS: default settings')
    download_ccds()
    if inSurveyccds:
      uniq_dir= 'against_surveyccds'
      img_patt= 'ksb_160711_*_ooi_*.fits.fz'
    else:
      uniq_dir= 'against_idl'
      raise ValueError('%s not supported yet' % uniq_dir)
      img_patt= 'k4m*170221_072831*ooi*.fits.fz'
    if ps1_only:
      ps1_gaia_dir= 'ps1_only'
      extra_args= ['--ps1_only']
    else:
      ps1_gaia_dir= 'ps1_gaia'
      extra_args= []
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','90prime',
                          ps1_gaia_dir,uniq_dir)
    fns= glob( os.path.join(os.path.dirname(__file__),
                            'testdata','ccds_90prime',
                            img_patt))
    assert(len(fns) > 0)
    cmd_line=['--camera', '90prime','--outdir', outdir, 
              '--not_on_proj','--debug'] + PS1_GAIA_ARGS + extra_args
    run_and_check_outputs(image_list=fns, cmd_line=cmd_line,
                          outdir=outdir)
    #run_and_check_outputs(image_list=[fns[0]], cmd_line=cmd_line,
    #                      outdir=outdir)
    

if __name__ == "__main__":
  # Run on images, compare to IDL zeropoints
  #test_decam(inSurveyccds=False, ps1_only=False)
  #test_mosaic(inSurveyccds=False, ps1_only=False)
  
  # Run on images, compare to survey-ccds
  #test_decam(inSurveyccds=True, ps1_only=False)
  #test_mosaic(inSurveyccds=True, ps1_only=False)
  test_90prime(inSurveyccds=True, ps1_only=False)
 

