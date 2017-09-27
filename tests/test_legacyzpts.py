import os
from glob import glob

from legacyzpts.legacy_zeropoints import main,get_parser
from legacyzpts.fetch import fetch_targz

DOWNLOAD_DIR='http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'

CAMERAS= ['decam','mosaic','90prime']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "90prime":"bs4"}

def download_ccds():
    outdir= os.path.join(os.path.dirname(__file__),
                         'testdata')
    fetch_targz(os.path.join(DOWNLOAD_DIR,
                             'ccds.tar.gz'), 
                outdir)
    fetch_targz(os.path.join(DOWNLOAD_DIR,
                             'ccds_mosaic.tar.gz'), 
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


def test_decam_ps1_gaia():
    """Runs at least 1 CCD per band
    """
    print('RUNNING LEGACYZPTS: default settings')
    download_ccds()
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','new_legacyzpts_data',
                          'ps1_gaia')
    fns= glob( os.path.join(os.path.dirname(__file__),
                            'testdata','ccds',
                            'small_c4d_*oki_*_v1.fits.fz'))
    cmd_line=['--camera', 'decam','--outdir', outdir, 
              '--not_on_proj', '--debug']
    run_and_check_outputs(image_list=fns, cmd_line=cmd_line,
                          outdir=outdir)
    #run_and_check_outputs(image_list=[fns[0]], cmd_line=cmd_line,
    #                      outdir=outdir)

def test_decam_ps1_only():
    """Runs at least 1 CCD per band
    """
    print('RUNNING LEGACYZPTS: ps1_only')
    download_ccds()
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','new_legacyzpts_data',
                          'ps1_only')
    fns= glob( os.path.join(os.path.dirname(__file__),
                            'testdata','ccds',
                            'small_c4d_*oki_*_v1.fits.fz'))
    cmd_line=['--camera', 'decam','--outdir', outdir, 
              '--not_on_proj', '--debug', '--ps1_only']
    run_and_check_outputs(image_list=fns, cmd_line=cmd_line,
                          outdir=outdir)
    #run_and_check_outputs(image_list=[fns[0]], cmd_line=cmd_line,
    #                      outdir=outdir)

def test_mosaic_ps1_gaia():
    """Runs at least 1 CCD per band
    """
    print('RUNNING LEGACYZPTS: default settings')
    download_ccds()
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','mosaic',
                          'ps1_gaia')
    fns= glob( os.path.join(os.path.dirname(__file__),
                            'testdata','ccds_mosaic',
                            'k4m*ooi*.fits.fz'))
    cmd_line=['--camera', 'mosaic','--outdir', outdir, 
              '--not_on_proj','--debug']
    run_and_check_outputs(image_list=fns, cmd_line=cmd_line,
                          outdir=outdir)
    #run_and_check_outputs(image_list=[fns[0]], cmd_line=cmd_line,
    #                      outdir=outdir)




if __name__ == "__main__":
  #test_decam_ps1_gaia()
  #test_decam_ps1_only()
  test_mosaic_ps1_gaia()

