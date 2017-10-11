import matplotlib.pyplot as plt
import numpy as np 
import os
from collections import defaultdict

from legacyzpts.qa.paper_plots import myscatter

CAMERAS= ['decam','mosaic','90prime']
PRODUCTS= ['legacypipe','zpt','star-photom','star-astrom']

def get_tolerance(camera=None,
                  legacyzpts_product=None):
  """Returns dict giving maximum 'plus/minus' difference for each numeric column
  
  Note, dont fill in if differnce should be zero, that's default

  Args:
    camera: CAMERAS
    legacyzpts_product: legacypipe,zpt,star-photom,star-astrom
      arjuns idl table have inconsistant units so need to know which table considering
      e.g. idl decam zeropoint- and surveyccds- tables: fwhm,seeing are pixels, 
      but mosaic tables zeropoint- has fwhm,seeing arcsec while surveyccds- has units pixels
  """
  assert(camera in CAMERAS)
  assert(legacyzpts_product in PRODUCTS)
  pix_scale= {"mosaic":0.260,
              "decam":0.262,
              "90prime":0.470}[camera]
  pm={}
  if camera == 'decam':
    # legacypipe keys
    for key in ['ccdzpt']: #cczpt strict
      pm[key]= 0.007
    for key in ['zpt']: # Average ccdzpt not as strict
      pm[key]= 0.1
    for key in ['ra','dec','ra_bore','dec_bore']:
      pm[key]= 0.5/3600 
    for key in ['ccdraoff','ccddecoff']:
      pm[key]= 0.15 
    for key in ['ccdnmatch']:
      pm[key]= 150
    for key in ['fwhm', 'seeing']:
      pm[key]= 0.3/pix_scale # Pixels is default unit
      if key == 'seeing': # idl units are arcsec
        pm[key] *= pix_scale
    # zpt keys
    for key in ['ccdskycounts','ccdskyrms']:
      pm[key]= 1.
    for key in ['avsky']:
      pm[key]= 1.e-4 # a header value
    for key in ['ccdphoff']:
      pm[key]= 0.35
    for key in ['ccdphrms']:
      pm[key]= 0.08
    for key in ['ccdskymag']:
      pm[key]= 0.1
    for key in ['ccdtransp']:
      pm[key]= 0.1
    for key in ['ccdra','ccddec']:
      pm[key]= 0.5/3600 
    for key in ['ccdrarms', 'ccddecrms']:
      pm[key]= 0.05 
    for key in ['cd1_1','cd1_2','cd2_1', 'cd2_2']:
      pm[key]= 1.e-10 

  elif camera == 'mosaic':
    # legacypipe keys
    for key in ['ccdzpt']: #cczpt strict
      pm[key]= 0.004
    for key in ['zpt']: # Average ccdzpt not as strict
      pm[key]= 0.03
    for key in ['ra','dec','ra_bore','dec_bore']:
      pm[key]= 0.5/3600 
    for key in ['ccdraoff','ccddecoff']:
      pm[key]= 10e-3 #arcsec
    for key in ['ccdnmatch']:
      pm[key]= 50
    for key in ['fwhm', 'seeing']:
      pm[key]= 0.1/pix_scale # Pixels is default unit
      if legacyzpts_product == 'zpt': # idl uses arcsec
        pm[key] *= pix_scale
    # zpt keys
    for key in ['ccdskycounts','ccdskyrms']:
      pm[key]= 1.e-2
    for key in ['avsky']:
      pm[key]= 1.e-4 # a header value
    for key in ['ccdphoff']:
      pm[key]= 0.2
    for key in ['ccdphrms']:
      pm[key]= 0.05
    for key in ['ccdskymag']:
      pm[key]= 0.15
    for key in ['ccdtransp']:
      pm[key]= 1.e-2
    for key in ['ccdra','ccddec']:
      pm[key]= 0.5/3600 
    for key in ['ccdrarms', 'ccddecrms']:
      pm[key]= 70./3600 
    for key in ['cd1_1','cd1_2','cd2_1', 'cd2_2']:
      pm[key]= 1.e-10 

  elif camera == '90prime':
    # legacypipe keys
    for key in ['ccdzpt']: #cczpt strict
      pm[key]= 0.004
    for key in ['zpt']: # Average ccdzpt not as strict
      pm[key]= 0.2 #0.03
    for key in ['ra','dec','ra_bore','dec_bore']:
      pm[key]= 1./3600 #deg
    for key in ['ccdraoff']:
      pm[key]= 40e-3 #arcsec
    for key in ['ccddecoff']:
      pm[key]= 20e-2 #arcsec
    for key in ['ccdnmatch']:
      pm[key]= 300
    for key in ['fwhm', 'seeing']:
      pm[key]= 0.2/pix_scale # Pixels is default unit
      if legacyzpts_product == 'zpt': # idl uses arcsec
        pm[key] *= pix_scale
    # zpt keys
    for key in ['ccdskycounts','ccdskyrms']:
      pm[key]= 1.e-2
    for key in ['avsky']:
      pm[key]= 1.e-4 # a header value
    for key in ['ccdphoff']:
      pm[key]= 0.004
    for key in ['ccdphrms']:
      pm[key]= 1.
    for key in ['ccdskymag']:
      pm[key]= 0.01
    for key in ['ccdtransp']:
      pm[key]= 0.01
    for key in ['ccdra','ccddec']:
      pm[key]= 0.6/3600 #10 mas
    for key in ['ccdrarms', 'ccddecrms']:
      pm[key]= 0.17 
    for key in ['cd1_1','cd1_2','cd2_1', 'cd2_2']:
      pm[key]= 1.e-10 
 

  return pm

def get_tolerance_star(camera=None,
                       legacyzpts_product=None):
  """Same as get_tolerance but for -star tables
  
  Note, dont fill in if differnce should be zero, that's default

  Args:
    camera: CAMERAS
    legacyzpts_product: legacypipe,zpt,star-photom,star-astrom
      arjuns idl table have inconsistant units so need to know which table considering
      e.g. idl decam zeropoint- and surveyccds- tables: fwhm,seeing are pixels, 
      but mosaic tables zeropoint- has fwhm,seeing arcsec while surveyccds- has units pixels
  """
  assert(camera in CAMERAS)
  assert(legacyzpts_product in PRODUCTS)
  pix_scale= {"mosaic":0.260,
              "decam":0.262}[camera]
  pm={}
  if camera == 'decam':
    # star keys
    for key in ['ccd_x','ccd_y']:
      pm[key]= 1.e-3
    for key in ['ccd_ra','ccd_dec']:
      pm[key]= 1.e-8
    for key in ['ccd_mag']:
      pm[key]= 1.
    for key in ['ccd_sky']:
      pm[key]= 10.
    for key in ['magoff']:
      pm[key]= 1.
    for key in ['raoff','decoff']:
      pm[key]= 1./3600 # arcsec

  elif camera == 'mosaic':
    # star keys
    for key in ['ccd_x','ccd_y']:
      pm[key]= 1.e-3
    for key in ['ccd_ra','ccd_dec']:
      pm[key]= 1.e-7
    for key in ['ccd_mag']:
      pm[key]= 0.5
    for key in ['ccd_sky']:
      pm[key]= 2.
    for key in ['magoff']:
      pm[key]= 0.5
    for key in ['raoff','decoff']:
      pm[key]= 1./3600 # arcsec
  
  return pm



def printDifference(col,data,ref,tol,
                    legacyzpts_product=None):
    """printer fro differenceChecker
    
    Args:
      tol: one of values of get_tolerance() 
    """
    assert(legacyzpts_product in PRODUCTS)
    abs_diff= np.abs(data.get(col) - ref.get(col))
    #if ('star' in legacyzpts_product):
    if len(data) > 10:
      print('col=',col,'data min,med,max= %g,%g,%g ref min,med,max= %g,%g,%g' % 
              (data.get(col).min(),np.median(data.get(col)),data.get(col).max(),
               ref.get(col).min(),np.median(ref.get(col)),ref.get(col).max()))
      print('\tpm=%g' % tol,'> abs_diff min,med,max= %g,%g,%g' % 
              (abs_diff.min(),np.median(abs_diff),abs_diff.max()))
    else:
      print('col=',col,'data=',data.get(col),'ref=',ref.get(col))
      print('\tpm=%g, abs_diff <=' % tol,abs_diff)
    return abs_diff

def differenceChecker(data,ref, cols, 
                      camera=None,legacyzpts_product=None):
  """Checks that difference between data and reference, for all columns, is small
  
  Args:
    data, ref: fits_tables of data and reference data
      example: data would be zpt table and ref would be idl zeropoint table
    cols: list of cols into data,ref
    camera: CAMERAS
    legacyzpts_product: legacypipe,zpt,star-photom,star-astrom
    nrows: number of rows to print so can see what differnces are
  """
  assert(camera in CAMERAS)
  assert(legacyzpts_product in PRODUCTS)
  for col in cols:
    print('will check col=%s' % col)
    assert(col in data.get_columns())
    assert(col in ref.get_columns())
  if 'star' in legacyzpts_product:
    pm= get_tolerance_star(camera,
                          legacyzpts_product=legacyzpts_product)
  else:
    pm= get_tolerance(camera,
                      legacyzpts_product=legacyzpts_product)
  for col in cols:
    #assert(np.all(np.isfinite(data.get(col))))
    abs_diff= printDifference(col,data,ref,
                              pm.get(col,0.), legacyzpts_product=legacyzpts_product)
    #print('col=',col,'data=',data.get(col),'ref=',ref.get(col))
    #print('\tpm=%g, abs_diff <=' % pm.get(col,0.),abs_diff)
    #print('have max(abs_diff)=',abs_diff.max(),'and tol=',pm.get(col,0.) )
    assert(np.all( abs_diff <= pm.get(col,0.) ))


def PlotDifference(legacyzpts_product='zpt',
                   camera=None,
                   indir='ps1_gaia',against='idl',prod=False,
                   x=None,y=None, cols=None,
                   xname='IDL',yname='Legacy'):
    """Plots y-x vs. x for all keys in tolerance dict pm

    Note: split by band and colored by ccdname

    Args:
      legacyzpts_product: PRODUCTS
      camera: CAMERAS
      indir: the testoutput directory to read from
      agaist: plotting against idl zeropoint or surveyccds
      prod: tests written to testoutput/ dir, if True it will look for production run
        outputs which are assumed to be copied to prodoutput/ dir
      x,y: astrometry.net fits_tables, they have same columns that will plot vs each other
      cols: cols in x,y fits_table to plot agains each other
      xname,yname: name for x and y data  
    """
    assert(legacyzpts_product in PRODUCTS)
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])
    assert(against in ['idl','surveyccds'])
    for col in cols:
      print('will plot col=%s' % col)
      assert(col in x.get_columns())
      assert(col in y.get_columns())
    if 'star' in legacyzpts_product:
      pm= get_tolerance_star(camera,
              legacyzpts_product=legacyzpts_product) 
    else:
      pm= get_tolerance(camera,
              legacyzpts_product=legacyzpts_product)
    # Plot
    FS=25
    eFS=FS+5
    tickFS=FS
    # In legacyzpts but potentially not idl
    if 'filter' in y.get_columns():
      bands= np.sort(list(set(y.filter)))
      band_arr= y.filter
      ccdname_arr= y.ccdname
    else:
      bands= np.sort(list(set(x.filter)))
      band_arr= x.filter
      ccdname_arr= x.ccdname
    testoutput= 'testoutput'
    if prod:
      testoutput= testoutput.replace('test','prod')
    save_dir= os.path.join(os.path.dirname(__file__),
                           testoutput,camera,
                           indir,'against_%s' % against)
    for cnt,col in enumerate(cols):
        # Plot
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        # Loop over bands, hdus
        for row,band in zip( range(3), bands ):
            hasBand= band_arr == band
            for ccdname,color in zip(set(ccdname_arr),['g','r','m','b','k','y']*12):
                keep= ((hasBand) &
                       (ccdname_arr == ccdname))
                if len(x[keep]) > 0:
                    xs= x.get(col)[keep]
                    ys= y.get(col)[keep] - xs
                    myscatter(ax[row], xs,ys, color=color,m='o',s=100,alpha=0.75) 
            # Label grz
            ax[row].text(0.9,0.9,band, 
                         transform=ax[row].transAxes,
                         fontsize=FS)
            ax[row].axhline(y=0.,xmin=xs.min(),xmax=xs.max(),c='k',lw=2.,ls='--') 
        supti= ax[0].set_title(col,fontsize=FS)
        xlab = ax[row].set_xlabel(xname,fontsize=FS)
        for row in range(3):
            ylab= ax[row].set_ylabel('%s - %s' % (yname,xname),fontsize=FS)
            ax[row].tick_params(axis='both', labelsize=tickFS)
            ax[row].set_ylim( (-pm[col],pm[col]) )
        savefn=os.path.join(save_dir,
                            '%s_%s.png' % (legacyzpts_product,col))
        plt.savefig(savefn, bbox_extra_artists=[supti,xlab,ylab], 
                    bbox_inches='tight')
        plt.close() 
        print("wrote %s" % savefn)


