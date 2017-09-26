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

def test_decam():
    """Runs a test image for each band of a given camera
    """
    print('Zeropoints for DECam')
    download_ccds()
    outdir = os.path.join(os.path.dirname(__file__),
                          'testoutput','ccds')
    fns= glob( os.path.join(os.path.dirname(__file__),
                            'testdata','ccds',
                            'small_c4d_*oki_*_v1.fits.fz'))
    assert(len(fns) > 0)
    cmd_line=['--camera', 'decam','--outdir', outdir, 
              '--not_on_proj', '--debug']
    parser= get_parser()
    args = parser.parse_args(args=cmd_line)
    main(image_list=fns, args=args)
    # check output files exits
    for fn in fns:
      base= os.path.basename(fn).replace(".fits.fz","")
      assert( os.path.exists(
                  os.path.join(outdir,
                               base+"-debug-zpt.fits")))
      assert( os.path.exists(
                  os.path.join(outdir,
                               base+"-debug-star.fits")))
      assert( os.path.exists(
                  os.path.join(outdir,
                               base+"-debug-legacypipe.fits")))


if __name__ == "__main__":
  test_decam()

