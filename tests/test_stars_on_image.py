import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Wedge
from matplotlib.collections import PatchCollection
import os
from glob import glob
import numpy as np
import fitsio

from astrometry.util.fits import fits_table, merge_tables


CAMERAS= ['decam','mosaic','90prime']

FN_SUFFIX= {"decam":"c4d",
            "mosaic": "k4m",
            "90prime":"bs4"}


def imgs2fits(images,name):
    '''images -- list of numpy 2D arrays'''
    assert('.fits' in name)
    hdu = fitsio.FITS(name,'rw')
    for image in images:
        hdu.write(image)
    hdu.close()
    print('Wrote %s' % name)

#def run_imshow_stars(search,arjun_fn):
#    from glob import glob 
#    extra_fns= glob(search)
#    for fn in extra_fns:
#        #for xx1,xx2,yy1,yy2 in [(300,700,600,1000),
#        #                              (3200,3700,0,500),
#        #                              (2400,2800,0,500)]:
#        #    imshow_stars_on_ccds(fn,arjun_fn,img_or_badpix='img',
#        #                         xx1=xx1,xx2=xx2,yy1=yy1,yy2=yy2)
#        imshow_stars_on_ccds(fn,arjun_fn=arjun_fn, img_or_badpix='img')
#        #imshow_stars_on_ccds(fn,arjun_fn=None, img_or_badpix='badpix')
#    print('Finished run_imshow_stars')

def plot_image(ax,image):
    vmin=np.percentile(image,q=0.5);vmax=np.percentile(image,q=99.5)
    ax.imshow(image.T, interpolation='none', origin='lower',
              cmap='gray',vmin=vmin,vmax=vmax)
    ax.tick_params(direction='out')
 
def plot_xy(ax,xs=None,ys=None,
            color='y',r_pixels=3.5/0.262):
    """
    xs,ys: pixel positions, array like
    """
    dr= r_pixels/ 4
    # img transpose used, so reverse x,y
    #patches=[Wedge((x,y), r_pixels + dr, 0, 360,dr) 
    patches=[Wedge((y, x), r_pixels + dr, 0, 360,dr) 
             for x,y in zip(xs, ys) ]
    coll = PatchCollection(patches, color=color) #,alpha=1)
    ax.add_collection(coll)
           
def plot_xy_json(ax,xy_json_fn,camera=None):
    from legacyzpts.common import loadjson
    XY= loadjson(xy_json_fn)
    colors= ['y','m','b','w','r']
    assert(camera in ['decam','90prime','mosaic'])
    pix_scale= {"mosaic":0.260,
                "decam":0.262,
                "90prime":0.470}[camera]
    for key,color,r_arcsec in zip(['dao','photom'],colors,[7,14]):
      plot_xy(ax,xs=XY['%s_x' % key],ys=XY['%s_y' % key],
              color=color,r_pixels=r_arcsec/pix_scale)
      

def plot_ccd(camera,imgfn,ccdname,
             xy_json_fn=None,
             xx1=0,xx2=4096-1,yy1=0,yy2=2048-1,
             savedir='.'):
  """Imshow ccd, optionally add xy positions of sources

  Args:
   xy_json_fn: xy*.json file
   xx1,xx2,yy1,yy2: bounding box in pixels
  """
  fig,ax=plt.subplots(figsize=(20,10))
  hdu= fitsio.FITS(imgfn)
  image= hdu[ccdname].read()
  #
  plot_image(ax,image)
  if xy_json_fn:
    plot_xy_json(ax,xy_json_fn,camera=camera)
  #
  plt.xlim(xx1,xx2)
  plt.ylim(yy1,yy2)
  savefn= os.path.basename(imgfn).replace('.fits','').replace('.fz','')
  savefn= '%s_%s_x%d-%d_y%d-%d.png' % (savefn,ccdname,xx1,xx2,yy1,yy2)
  savefn= os.path.join(savedir,savefn)
  plt.savefig(savefn,dpi=200)
  plt.close()
  print('Wrote %s' % savefn)


def overplot_stars(star_photom_fn, camera=None,ccdname=None):
  """
  
  Args:
    star_photom_fn: like path/to/ksb_*-star-photom.fits
    ccdname: N4
  """
  assert(camera in CAMERAS)
  stars= fits_table(star_photom_fn)
  W,H= stars.width[0],stars.height[0]
  # img
  imgroot= os.path.basename(star_photom_fn).split('-')[0] 
  imgfn= os.path.join(os.path.dirname(__file__),
                      'testdata','ccds_%s' % camera,
                      imgroot + '.fits.fz')
  hdu= fitsio.FITS(imgfn)
  img= hdu[ccdname].read()
  imshow_stars(img,camera=camera,
               xs=stars.x,ys=stars.y,
               xx1=0,xx2=W-1,yy1=0,yy2=H-1,
               name='%s_%s' % (imgroot,ccdname))
  imshow_stars(img,camera=camera,
               xs=stars.x,ys=stars.y,
               xx1=0,xx2=500,yy1=0,yy2=500,
               name='%s_%s' % (imgroot,ccdname))


if __name__ == "__main__":
  camera= '90prime'
  ccdname='CCD1'
  indir= 'ps1_gaia'
  nums= "160711_103513"

  ccd_dir=  os.path.join(os.path.dirname(__file__),
                         'testdata','ccds_%s' % camera)
  zpts_dir= os.path.join(os.path.dirname(__file__),
                         'testoutput',camera,
                         indir,'against_surveyccds')
  expnum= {"160711_103513":"75800084",
           "160711_070206":"75800144"}

  imgfn= os.path.join(ccd_dir,
                      'ksb_%s_ooi_g_v1.fits.fz' % nums)
  xy_json_fn= os.path.join(zpts_dir,
                      'xy_%s_%s.json' % (expnum[nums],ccdname))
  zpts= fits_table(os.path.join(zpts_dir,
            'ksb_%s_ooi_g_v1-debug-legacypipe.fits' % nums))
  zpts.cut( np.char.strip(zpts.ccdname) == ccdname)
  W,H= zpts.width[0],zpts.height[0]

  plot_ccd(camera,imgfn,ccdname,
           xy_json_fn=xy_json_fn,
           xx1=0,xx2=W-1,yy1=0,yy2=H-1,
           #xx1=1000,xx2=2000,yy1=0,yy2=500,
           savedir=zpts_dir)
  
  #camera='90prime'
  #dr= 'tests/testoutput/%s/ps1_gaia/against_surveyccds/' % camera
  #fn= dr+ 'ksb_160711_070206_ooi_r_v1-debug-star-photom.fits'
  #test_overplot_stars(fn, camera=camera,ccdname='ccd1')
  
  
  #test_overplot_stars(fn, camera=camera,ccdname='ccd1')

