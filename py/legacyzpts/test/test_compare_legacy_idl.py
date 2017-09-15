from legacyzpts.qa.compare_idlzpts import ZptResiduals, StarResiduals
from legacyzpts.fetch import fetch_targz

def load_test_data():
    fetch_targz()

def test_zpts():
        leg_fns= glob(os.path.join(os.getenv['CSCRATCH'],
                                   'dr5_zpts/decam',
                                   'c4d*-zpt.fits')
        idl_fns= glob(os.path.join(os.getenv['CSCRATCH'],
                                   'arjundey_Test/AD_exact_skymed',
                                   'zeropoint*.fits')
        zpt= ZptResiduals(camera='decam',savedir='.',
                          leg_list=leg_list,
                          idl_list=idl_list)
        zpt.load_data()
        zpt.convert_legacy()
        zpt.match(ra_key='ccdra',dec_key='ccdec')
        zpt.plot_residuals(doplot='diff') 

    
def test_stars():
  leg_fns= glob(os.path.join(os.getenv['CSCRATCH'],
                             'dr5_zpts/decam',
                             'c4d*-star.fits')
  idl_fns= glob(os.path.join(os.getenv['CSCRATCH'],
                             'arjundey_Test/AD_exact_skymed',
                             'matches*.fits')
  star= StarResiduals(camera='decam',savedir='.',
                      leg_list=leg_list,
                      idl_list=idl_list)
  star.load_data()
  star.convert_legacy()
  star.match(ra_key='ccd_ra',dec_key='ccd_dec')
  star.plot_residuals(doplot='diff') 
