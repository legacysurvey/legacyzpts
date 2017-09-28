import matplotlib.pyplot as plt
import numpy as np 
import os

from legacyzpts.qa.paper_plots import myscatter

CAMERAS= ['decam','mosaic','bok']

def get_tolerance(camera=None):
  """Returns dict giving maximum 'plus/minus' difference for each numeric column
  
  Note, dont fill in if differnce should be zero, that's default
  """
  pix= {"mosaic":0.260,
        "decam":0.262}
  pm={}
  if camera == 'decam':
    # legacypipe keys
    for key in ['zpt','ccdzpt']:
      pm[key]= 8.e-3 #8 mmag
    for key in ['ra','dec','ra_bore','dec_bore']:
      pm[key]= 0.5/3600 #10 mas
    for key in ['ccdraoff','ccddecoff']:
      pm[key]= 60e-3 # arcsec
    for key in ['ccdnmatch']:
      pm[key]= 80
    for key in ['fwhm']:
      pm[key]= 0.1/pix[camera] #0.5 as
    # zpt keys
    for key in ['ccdskycounts']:
      pm[key]= 0.1

  elif camera == 'mosaic':
    # legacypipe keys
    for key in ['zpt','ccdzpt']:
      pm[key]= 8.e-3 #8 mmag
    for key in ['ra','dec','ra_bore','dec_bore']:
      pm[key]= 0.5/3600 #10 mas
    for key in ['ccdraoff','ccddecoff']:
      pm[key]= 10e-3 #arcsec
    for key in ['ccdnmatch']:
      pm[key]= 30
    for key in ['fwhm']:
      pm[key]= 0.1/pix[camera] #0.5 as
    # zpt keys
    for key in ['ccdskycounts']:
      pm[key]= 0.1
  
  return pm

def differenceChecker(data,ref, cols, camera=None):
  """Checks that difference between data and reference, for all columns, is small
  
  Args:
    data, ref: fits_tables of data and reference data
      example: data would be zpt table and ref would be idl zeropoint table
    cols: list of cols into data,ref
  """
  assert(camera in CAMERAS)
  for col in cols:
    print('will check col=%s' % col)
    assert(col in data.get_columns())
    assert(col in ref.get_columns())
  pm= get_tolerance(camera)
  for col in cols:
    assert(np.all(np.isfinite(data.get(col))))
    abs_diff= np.abs(data.get(col) - ref.get(col))
    print('col=',col,'data=',data.get(col),'ref=',ref.get(col))
    print('\tpm=%g, abs_diff <=' % pm.get(col,0.),abs_diff)
    assert(np.all( abs_diff <= pm.get(col,0.) ))


def PlotDifference(camera=None,indir='ps1_gaia',against='idl',
                   which_table='zpt',
                   x=None,y=None, cols=None,
                   xname='IDL',yname='Legacy'):
    """Plots y-x vs. x for all keys in tolerance dict pm

    Note: split by band and colored by ccdname

    Args:
      camera: CAMERAS
      indir: the testoutput directory to read from
      agaist: plotting against idl zeropoint or surveyccds
      which_table: either zpt,star,legacypipe
      x,y: astrometry.net fits_tables, they have same columns that will plot vs each other
      cols: cols in x,y fits_table to plot agains each other
      xname,yname: name for x and y data  
    """
    assert(camera in CAMERAS)
    assert(indir in ['ps1_gaia','ps1_only'])
    assert(against in ['idl','surveyccds'])
    assert(which_table in ['zpt','star','legacypipe'])
    for col in cols:
      print('will plot col=%s' % col)
      assert(col in x.get_columns())
      assert(col in y.get_columns())
    pm= get_tolerance(camera)
    # Plot
    FS=25
    eFS=FS+5
    tickFS=FS
    bands= np.sort(list(set(x.filter)))
    ccdnames= set(x.ccdname)
    save_dir= os.path.join(os.path.dirname(__file__),
                           'testoutput',camera,
                           indir,'against_%s' % against)
    for cnt,col in enumerate(cols):
        # Plot
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        # Loop over bands, hdus
        for row,band in zip( range(3), bands ):
            hasBand= x.filter == band
            for ccdname,color in zip(ccdnames,['g','r','m','b','k','y']*12):
                keep= ((hasBand) &
                       (x.ccdname == ccdname))
                if len(x[keep]) > 0:
                    xs= x.get(col)[keep]
                    ys= y.get(col)[keep] - xs
                    myscatter(ax[row], xs,ys, color=color,m='o',s=100,alpha=0.75) 
            # Label grz
            ax[row].text(0.9,0.9,band, 
                         transform=ax[row].transAxes,
                         fontsize=FS)
        supti= ax[0].set_title(col,fontsize=FS)
        xlab = ax[row].set_xlabel(xname,fontsize=FS)
        for row in range(3):
            ylab= ax[row].set_ylabel('%s - %s' % (yname,xname),fontsize=FS)
            ax[row].tick_params(axis='both', labelsize=tickFS)
            ax[row].set_ylim( (-pm[col],pm[col]) )
        savefn=os.path.join(save_dir,
                            '%s_%s.png' % (which_table,col))
        plt.savefig(savefn, bbox_extra_artists=[supti,xlab,ylab], 
                    bbox_inches='tight')
        plt.close() 
        print("wrote %s" % savefn)


