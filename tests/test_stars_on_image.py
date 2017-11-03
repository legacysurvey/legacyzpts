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

def victory_plot_bass_ccd2():
    zpt= lambda ps1mag_minus_apmag: np.median(sigmaclip(ps1mag_minus_apmag, low=2.5, high=2.5)[0])
    hdu = fits.open('ksb_160711_070206_ooi_r_v1_CCD2_all_stars.fits')
    a=hdu[1].data

    i= (a['is_iso']) & (a['good_flux_and_mag']) & (a['no_badpix_in_ap_0'])
    plt.scatter(a['ps1_mag'][i],a['ps1_mag'][i]-a['apmag'][i],c='g')
    i= (a['is_iso']) & (a['good_flux_and_mag']) & (a['no_badpix_in_ap_0_5']) & (a['no_badpix_in_ap_0'] == False)
    plt.scatter(a['ps1_mag'][i],a['ps1_mag'][i]-a['apmag'][i],c='k')
    i= (a['is_iso']) & (a['good_flux_and_mag']) & (a['no_badpix_in_ap_0'])
    val=zpt(a['ps1_mag'][i]-a['apmag'][i])
    plt.axhline(val,c='g',ls='--',label='%.2f (reject 5)' % val)
    i= (a['is_iso']) & (a['good_flux_and_mag']) & (a['no_badpix_in_ap_0_5']) & (a['no_badpix_in_ap_0'] == False)
    val=zpt(a['ps1_mag'][i]-a['apmag'][i])
    plt.axhline(val,c='k',ls='--',label='%.2f (keep 5)' % val)
    val=-1.92
    plt.axhline(val,c='r',ls='--',label='%.2f (IDL ccdzpt - 27.01)' % val)
    plt.legend(loc='lower right')
    plt.xlabel('PS1 mag')
    plt.ylabel('PS1 - Legacyzpt')
    plt.title('ksb_160711_070206_ooi_r_v1.fits.fz, CCD2')
    plt.savefig('bam.png',dpi=200)


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
    colors= ['m','b','w','y','r']
    assert(camera in ['decam','90prime','mosaic'])
    pix_scale= {"mosaic":0.260,
                "decam":0.262,
                "90prime":0.470}[camera]
    # isolated if NN is >= 11 arcsec
    for key,color,r_arcsec in zip(['dao','hasbadpix','notiso','photom'],colors,[7.,10.,13.,16.]):
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
    extra=''
    if xy_json_fn:
        extra= 'xypts'
    savefn= '%s_%s_%s_x%d-%d_y%d-%d.png' % (savefn,ccdname,extra,xx1,xx2,yy1,yy2)
    savefn= os.path.join(savedir,savefn)
    plt.savefig(savefn,dpi=200)
    plt.close()
    print('Wrote %s' % savefn)

## Tests

def test_camera(camera='decam'):
    """Plots xy_dict stars on every test ccd for given camera"""
    assert(camera in CAMERAS)
    indir= 'ps1_gaia'
    ccds= {'decam':['N4','S4'],
         'mosaic':['CCD' + str(i) for i in [1,2,3,4]],
         '90prime':['CCD' + str(i) for i in [2]]}
    prefix= {'decam':'small_c4d',
           'mosaic':'k4m',
           '90prime':'ksb'}

    ccd_dir=  os.path.join(os.path.dirname(__file__),
                         'testdata','ccds_%s' % camera)
    zpts_dir= os.path.join(os.path.dirname(__file__),
                         'testoutput',camera,
                         indir,'against_surveyccds')

    patt= os.path.join(ccd_dir,
                '%s*ooi*.fits.fz' % (prefix[camera],) )
    print('patt=',patt)
    fns= glob(patt)
    for fn in fns:
        print('fn=%s' % fn)
        for ccdname in ccds[camera]:
            print('ccd=%s' % ccdname)
            xy_json_fn= os.path.join(zpts_dir,
                             os.path.basename(fn).replace('.fits.fz','_xy_%s.json' % ccdname))
            zpts_fn= os.path.join(zpts_dir,
                      os.path.basename(fn).replace('.fits.fz','-debug-legacypipe.fits'))
            if not os.path.exists(zpts_fn):
                print('skipping %s' % zpts_fn)
                continue
            zpts= fits_table(zpts_fn)
            zpts.cut( np.char.strip(zpts.ccdname) == ccdname)
            W,H= zpts.width[0],zpts.height[0]

            plot_ccd(camera,fn,ccdname,
                   xy_json_fn=None,
                   xx1=0,xx2=H-1,yy1=0,yy2=W-1,
                   #xx1=1000,xx2=2000,yy1=0,yy2=500,
                   savedir=zpts_dir)

            plot_ccd(camera,fn,ccdname,
                   xy_json_fn=xy_json_fn,
                   xx1=0,xx2=H-1,yy1=0,yy2=W-1,
                   #xx1=1000,xx2=2000,yy1=0,yy2=500,
                   savedir=zpts_dir)
      

if __name__ == "__main__":
    for camera in ['90prime']: #'decam','mosaic','90prime']:
        test_camera(camera)

