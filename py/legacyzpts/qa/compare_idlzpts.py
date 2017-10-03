'''
TO RUN:
idl vs. legacy comparison: python -c "from simulate_survey import Legacy_vs_IDL;a=Legacy_vs_IDL(camera='decam',leg_dir='/global/cscratch1/sd/kaylanb/kaylan_Test',idl_dir='/global/cscratch1/sd/kaylanb/arjundey_Test')"
idl vs. legacy number star matches: python -c "from simulate_survey import sn_not_matched_by_arjun;sn_not_matched_by_arjun('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-*-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')"
idl vs. legacy number star matches: python -c "from simulate_survey import number_matches_by_cut;number_matches_by_cut('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-*-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')"
oplot stars on ccd: python -c "from simulate_survey import run_imshow_stars;run_imshow_stars('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-35-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')"
'''

#if __name__ == "__main__":
#    import matplotlib
#    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import json
from glob import glob
#from sklearn.neighbors import KernelDensity
from collections import defaultdict
from scipy.stats import sigmaclip

#from PIL import Image, ImageDraw
import fitsio

try:
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.libkd.spherematch import match_radec
    from tractor.sfd import SFDMap
    from tractor.brightness import NanoMaggies
except ImportError:
    pass

from legacyzpts.qa import params
from legacyzpts.qa.paper_plots import LegacyZpts, myscatter
from legacyzpts.legacy_zeropoints import convert_zeropoints_table
from legacyzpts.legacy_zeropoints import convert_stars_table

mygray='0.6'

def sn_not_matched_by_arjun(extra_search,arjun_fn):
    from glob import glob 
    from astrometry.libkd.spherematch import match_radec
    # Load arjun's stars
    arj=fits_table(arjun_fn)
    hdu_to_ccdname= {'35':'N4 ','28':'S4 ','50':'N19','10':'S22'}
    # Loop over extra.pkl files
    extra_fns= glob(extra_search)
    for fn in extra_fns:
        with open(fn,'r') as foo:
            extra= pickle.load(foo)
        keep= arj.extname == hdu_to_ccdname[str(extra['hdu'])]
        # daofind
        m1, m2, d12 = match_radec(extra['ra'],extra['dec'],arj.ccd_ra,arj.ccd_dec,1./3600.0,nearest=True)
        not_m1= np.ones(len(extra['ra']),bool)
        not_m1[m1]=False
        print('ccdname %s, hdu %d' % (hdu_to_ccdname[str(extra['hdu'])],extra['hdu']))
        print('Arjun doesnt find %d stars that I do' % np.where(not_m1)[0].size)
        sn= extra['apflux'] / np.sqrt(extra['apskyflux'])
        print('These stars have: x,y,SN WHERE X,Y ARE HORIZ,VERT')
        for x,y,s in zip(extra['x'][not_m1],extra['y'][not_m1],sn[not_m1]):
            print('%d, %d, %.1f' % (x,y,s))
            



def number_matches_by_cut(extra_search,arjun_fn):
    # Load arjun's stars
    arj=fits_table(arjun_fn)
    hdu_to_ccdname= {'35':'N4 ','28':'S4 ','50':'N19','10':'S22'}
    # Loop over extra.pkl files
    extra_fns= glob(extra_search)
    for fn in extra_fns:
        with open(fn,'r') as foo:
            extra= pickle.load(foo)
        keep= arj.extname == hdu_to_ccdname[str(extra['hdu'])]
        print('ccdname %s' % hdu_to_ccdname[str(extra['hdu'])])
        # daofind
        m1, m2, d12 = match_radec(extra['dao_ra'],extra['dao_dec'],arj.ccd_ra,arj.ccd_dec,1./3600.0,nearest=True)
        print('daofind', len(m1),len(arj[keep]))
        # cleaning cuts
        for key in ['apmags','apflux','b_isolated','separation','bit_flux']:
            cut=extra[key]
            m1, m2, d12 = match_radec(extra['dao_ra'][cut],extra['dao_dec'][cut],arj.ccd_ra,arj.ccd_dec,1./3600.0,nearest=True)
            print(key, len(m1),len(arj[keep]))
        # gaia match
        m1, m2, d12 = match_radec(extra['ra'],extra['dec'],arj.ccd_ra,arj.ccd_dec,1./3600.0,nearest=True) 
        print('gaia match', len(m1),len(arj[keep]))


def imgs2fits(images,name):
    '''images -- list of numpy 2D arrays'''
    assert('.fits' in name)
    hdu = fitsio.FITS(name,'rw')
    for image in images:
        hdu.write(image)
    hdu.close()
    print('Wrote %s' % name)

def imshow_stars(ccd_image,camera=None,
                 xs=None,ys=None,
                 xx1=0,xx2=4096-1,yy1=0,yy2=2048-1,
                 name='stars_on_ccd'):
    """Overlays positions on a CCD image
     
    Args:
     ccd_image: hdu[ccdname].read() array
     camera: CAMERAS
     xs,ys: x and y pixel positions
     xx1,xx2,yy1,yy2: bounding box in pixels
    """
    assert(camera in ['decam','90prime','mosaic'])
    pix_scale= {"mosaic":0.260,
                "decam":0.262,
                "90prime":0.470}[camera]
    from matplotlib.patches import Circle,Wedge
    from matplotlib.collections import PatchCollection
    fig,ax=plt.subplots(figsize=(20,10))
    vmin=np.percentile(ccd_image,q=0.5);vmax=np.percentile(ccd_image,q=99.5)
    ax.imshow(ccd_image.T, interpolation='none', origin='lower',
              cmap='gray',vmin=vmin,vmax=vmax)
    ax.tick_params(direction='out')
    aprad_pix= 10/2./ pix_scale
    drad= aprad_pix / 4
    #aprad_pix= 100.
    #for x,y,color,r in [(extra['daofind_x'],extra['daofind_y'],'y',aprad_pix),
    #                  (extra['mycuts_x'],extra['mycuts_y'],'b',aprad_pix + 1.25*drad),
    #                  (extra['x'],extra['y'],'m',aprad_pix + 2.5*drad)]:
    #iso= extra['b_isolated']
    #m1, m2, d12 = match_radec(extra['dao_ra'][iso],extra['dao_dec'][iso],extra['arjun_ra'],extra['arjun_dec'],1./3600.0,nearest=True)
    for x,y,color,r in [(xs,ys,'y',aprad_pix)]:
    #for x,y,color,r in [(extra['1st_x'],extra['1st_y'],'y',aprad_pix),
    #                    (extra['2nd_x'],extra['2nd_y'],'m',2*aprad_pix)]:
    #key2,key3 = 'apflux','bit_flux' 
    #key2,key3 = 'b_isolated','apmags' 
    #key2,key3 = 'b_isolated','separation' 
    #for x,y,color,r in [(extra['dao_x_5'],extra['dao_y_5'],'y',aprad_pix),
    #                    (extra['dao_x_5'][extra[key2]],extra['dao_y_5'][extra[key2]],'b',aprad_pix + 1.25*drad),
    #                    (extra['dao_x_5'][extra[key3]],extra['dao_y_5'][extra[key3]],'m',aprad_pix + 2.5*drad)]:
        # img transpose used, so reverse x,y
        #circles=[ Circle((y1, x1), rad) for x1, y1 in zip(x, y) ]
        patches=[ Wedge((y1, x1), r + drad, 0, 360,drad) for x1, y1 in zip(x, y) ]
        coll = PatchCollection(patches, color=color) #,alpha=1)
        ax.add_collection(coll)
        #plt.scatter(y,x,facecolors='none',edgecolors=color,marker='o',s=rad,linewidth=5.,rasterized=True)
    plt.xlim(xx1,xx2)
    plt.ylim(yy1,yy2)
    savefn= '%s_x%d-%d_y%d-%d.png' % (name,xx1,xx2,yy1,yy2)
    plt.savefig(savefn)
    plt.close()
    print('Wrote %s' % savefn)



def imshow_stars_on_ccds(extra_fn, arjun_fn=None,
                         xx1=0,xx2=4096-1,yy1=0,yy2=2048-1,
                         img_or_badpix='img'):
    assert(img_or_badpix in ['img','badpix'])
    from matplotlib.patches import Circle,Wedge
    from matplotlib.collections import PatchCollection
    fig,ax=plt.subplots(figsize=(20,10))
    with open(extra_fn,'r') as foo:
        extra= pickle.load(foo)
    if img_or_badpix == 'img':
        img= fitsio.read(extra['proj_fn'], ext=extra['hdu'], header=False)
        vmin=np.percentile(img,q=0.5);vmax=np.percentile(img,q=99.5)
    elif img_or_badpix == 'badpix':
        img= fitsio.read(extra['proj_fn'].replace('oki','ood').replace('ooi','ood'),
                         ext=extra['hdu'], header=False)
        vmin,vmax = None,None
        img[img > 0] = 1
        img[img == 0] = 2
        img[img == 1] = 0
        img[img == 2] = 1
    # Load arjun's stars
    if arjun_fn:
        #assert(hdu) #must be specified to match to ccdname
        arjun=fits_table(arjun_fn)
        hdu_to_ccdname= {'35':'N4 ','28':'S4 ','50':'N19','10':'S22'}
        #hdu_to_ccdname= {'34':'N4 ','27':'S4 ','49':'N19','9':'S22'}
        keep= arjun.extname == hdu_to_ccdname[str(extra['hdu'])]
        extra['arjun_x']= arjun.ccd_x[keep]
        extra['arjun_y']= arjun.ccd_y[keep]
        extra['arjun_ra']= arjun.ccd_ra[keep]
        extra['arjun_dec']= arjun.ccd_dec[keep]
    ax.imshow(img.T, interpolation='none', origin='lower',cmap='gray',vmin=vmin,vmax=vmax)
    ax.tick_params(direction='out')
    aprad_pix= 10/2./0.262
    drad= aprad_pix / 4
    #aprad_pix= 100.
    #for x,y,color,r in [(extra['daofind_x'],extra['daofind_y'],'y',aprad_pix),
    #                  (extra['mycuts_x'],extra['mycuts_y'],'b',aprad_pix + 1.25*drad),
    #                  (extra['x'],extra['y'],'m',aprad_pix + 2.5*drad)]:
    #iso= extra['b_isolated']
    #m1, m2, d12 = match_radec(extra['dao_ra'][iso],extra['dao_dec'][iso],extra['arjun_ra'],extra['arjun_dec'],1./3600.0,nearest=True)
    for x,y,color,r in [(extra['arjun_x'],extra['arjun_y'],'y',aprad_pix),
                        (extra['x'],extra['y'],'m',2*aprad_pix)]:
    #for x,y,color,r in [(extra['1st_x'],extra['1st_y'],'y',aprad_pix),
    #                    (extra['2nd_x'],extra['2nd_y'],'m',2*aprad_pix)]:
    #key2,key3 = 'apflux','bit_flux' 
    #key2,key3 = 'b_isolated','apmags' 
    #key2,key3 = 'b_isolated','separation' 
    #for x,y,color,r in [(extra['dao_x_5'],extra['dao_y_5'],'y',aprad_pix),
    #                    (extra['dao_x_5'][extra[key2]],extra['dao_y_5'][extra[key2]],'b',aprad_pix + 1.25*drad),
    #                    (extra['dao_x_5'][extra[key3]],extra['dao_y_5'][extra[key3]],'m',aprad_pix + 2.5*drad)]:
        # img transpose used, so reverse x,y
        #circles=[ Circle((y1, x1), rad) for x1, y1 in zip(x, y) ]
        patches=[ Wedge((y1, x1), r + drad, 0, 360,drad) for x1, y1 in zip(x, y) ]
        coll = PatchCollection(patches, color=color) #,alpha=1)
        ax.add_collection(coll)
        #plt.scatter(y,x,facecolors='none',edgecolors=color,marker='o',s=rad,linewidth=5.,rasterized=True)
    plt.xlim(xx1,xx2)
    plt.ylim(yy1,yy2)
    savefn= '%s_%s_x%d-%d_y%d-%d.png' % (img_or_badpix,extra['hdu'],xx1,xx2,yy1,yy2)
    #savefn= '%s_%s.png' % (img_or_badpix,extra['hdu'])
    plt.savefig(savefn)
    plt.close()
    print('Wrote %s' % savefn)

def run_imshow_stars(search,arjun_fn):
    from glob import glob 
    extra_fns= glob(search)
    for fn in extra_fns:
        #for xx1,xx2,yy1,yy2 in [(300,700,600,1000),
        #                              (3200,3700,0,500),
        #                              (2400,2800,0,500)]:
        #    imshow_stars_on_ccds(fn,arjun_fn,img_or_badpix='img',
        #                         xx1=xx1,xx2=xx2,yy1=yy1,yy2=yy2)
        imshow_stars_on_ccds(fn,arjun_fn=arjun_fn, img_or_badpix='img')
        #imshow_stars_on_ccds(fn,arjun_fn=None, img_or_badpix='badpix')
    print('Finished run_imshow_stars')

def stellarlocus(x,a,b,c,d,e):
    '''
    Arjun's stellarlocus2.pro fitting to I-W1, R-I
    ; Inputs:  
    ;   x - I-[3.6] color 
    ;   par - fit parameters: 
    ;       par0 + par1*x + par2*x^2 - par3/(1 + exp(par4*(x-par5))) 
    ; 
    ; Outputs: function outputs the R-I color 
    '''
    #return a + b/(1. + (1./c -1.)*np.exp(-d*x))
    return a + b*x + c*x**2 - d/(1. + np.exp(-d*(x-e)))

class Legacy_vs_IDL(object):
    def __init__(self,camera='decam',savedir='./',
                 zpts_or_stars='zpts',
                 leg_list=[],
                 idl_list=[]):
        assert(zpts_or_stars in ['zpts','stars'])
        self.camera= camera
        self.zpts_or_stars= zpts_or_stars
        self.savedir= savedir
        self.idl= LegacyZpts(zpt_list=idl_list, camera=camera, savedir=savedir)
        self.legacy= LegacyZpts(zpt_list=leg_list, camera=camera,savedir=savedir)

    def run(self):
        self.idl.load()
        self.legacy.load()
        # Convert legacy names/units to idl
        if False:
            from legacyzpts.legacy_zeropoints import legacy2idl_zpts
            self.legacy.data= legacy2idl_zpts(self.legacy.data)
            #ZeropointResiduals(object)
            #MatchesResiduals(object):
        self.match_zpts()

    def get_defaultdict_ylim(self,doplot,ylim=None):
        ylim_dict=defaultdict(lambda: ylim)
        if doplot == 'diff':
            ylim_dict['ra']= (-1e-8,1e-8)
            ylim_dict['dec']= ylim_dict['ra']
            ylim_dict['zpt']= (-0.005,0.005)
        elif doplot == 'div':
            ylim_dict['zpt']= (0.995,1.005)
            ylim_dict['skycounts']= (0.99,1.01)
        else:
            pass
        return ylim_dict     

    def plots_like_arjuns(self):
        # Plot
        FS=15
        eFS=FS+5
        tickFS=FS
        for image_fn in set(np.char.strip(self.legacy.stars.image_filename)):
            for hdu in set(self.legacy.stars.image_hdu):
                istar= (np.char.strip(self.legacy.stars.image_filename) == image_fn)*\
                       (self.legacy.stars.image_hdu == hdu)
                idata= (np.char.strip(self.legacy.data.image_filename) == image_fn)*\
                       (self.legacy.data.image_hdu == hdu)
                assert(len(self.legacy.data[idata]) == 1)
                base_image_fn= os.path.basename(image_fn).replace('.fits','')
                base_image_fn= base_image_fn.replace('.fz','')
                ccdname= self.legacy.data.ccdname[idata][0].strip()
                fig,ax= plt.subplots(2,2,figsize=(10,10))
                plt.subplots_adjust(hspace=0.3,wspace=0.3)
                # 1st panel: deciff vs. radiff
                myscatter(ax[0,0],self.legacy.stars.radiff[istar],self.legacy.stars.decdiff[istar],
                          color='b',m='o',s=50,alpha=0.75) 
                #ramed= np.median(self.legacy.stars.radiff[istar])
                ramed= self.legacy.data.raoff[idata]
                #rarms, _, _ = sigmaclip(self.legacy.stars.radiff[istar], low=3., high=3.)
                #rarms = getrms(rarms)
                rarms= self.legacy.data.rarms[idata]
                ax[0,0].axvline(0,color='k',linestyle='solid',linewidth=1)
                ax[0,0].axvline(ramed,color='k',linestyle='dashed',linewidth=1)
                #decmed= np.median(self.legacy.stars.decdiff[istar])
                decmed= self.legacy.data.decoff[idata]
                #decrms, _, _ = sigmaclip(self.legacy.stars.decdiff[istar], low=3., high=3.)
                #decrms = getrms(decrms)
                decrms = self.legacy.data.decrms[idata]
                ax[0,0].axhline(0,color='k',linestyle='solid',linewidth=1)
                ax[0,0].axhline(decmed,color='k',linestyle='dashed',linewidth=1)
                mytext(ax[0,0],0.05,0.95,'Median: %.3f,%.3f' % (ramed,decmed), 
                       fontsize=FS)
                mytext(ax[0,0],0.05,0.87,'RMS: %.3f,%.3f' % (rarms,decrms), 
                       fontsize=FS)
                ax[0,0].set_title('%s' % base_image_fn,fontsize=FS) 
                ax[0,0].set_xlabel('delta RA',fontsize=FS)
                ax[0,0].set_ylabel('delta DEC',fontsize=FS)
                ax[0,0].set_xlim(-1,1)
                ax[0,0].set_ylim(-1,1)
                # 2nd panel: (PS_band_mag + colorterm2decam) - (our apmag + zpt)
                temp_b= self.legacy.data.filter[idata]
                temp_zp= self.legacy.data.zpt[idata]
                temp_gain= self.legacy.data.gain[idata]
                assert(len(set(temp_b)) == 1)
                assert(len(set(temp_zp)) == 1)
                zpt_adu_per_sec= temp_zp[0] - (self.fid.zp0[temp_b[0]] - 2.5*np.log10(temp_gain[0]))
                myscatter(ax[0,1],self.legacy.stars.ps1_mag[istar],
                          self.legacy.stars.ps1_mag[istar] - (self.legacy.stars.apmag[istar]+zpt_adu_per_sec),
                          color='b',m='o',s=50,alpha=0.75) 
                ax[0,1].axhline(0,color='k',linestyle='dashed',linewidth=1)
                mytext(ax[0,1],0.05,0.95,
                       'Number of matches: %d' % self.legacy.data.nmatch[idata], 
                       fontsize=FS)
                mytext(ax[0,1],0.05,0.87,
                       'Zeropoint: %.3f' % self.legacy.data.zpt[idata], 
                       fontsize=FS)
                mytext(ax[0,1],0.05,0.81,
                       'Offset: %.3f' % self.legacy.data.phoff[idata], 
                       fontsize=FS)
                mytext(ax[0,1],0.05,0.25,
                       'RMS: %.3f' % self.legacy.data.phrms[idata], 
                       fontsize=FS)
                ax[0,1].set_title('%s' % self.legacy.data.expid[idata],fontsize=FS)
                ax[0,1].set_xlabel('PS1 + CT (Mag)',fontsize=FS)
                ax[0,1].set_ylabel('(PS1 + CT) - (DECam + ZP)',fontsize=FS)
                ax[0,1].set_xlim(14,22)
                ax[0,1].set_ylim(-0.5,0.5)
                # 3rd panel: 2nd but vs. g-i color
                gi= self.legacy.stars.ps1_g[istar] - self.legacy.stars.ps1_i[istar]
                myscatter(ax[1,0],gi,
                          self.legacy.stars.ps1_mag[istar] - (self.legacy.stars.apmag[istar]+zpt_adu_per_sec),
                          color='b',m='o',s=50,alpha=0.75) 
                ax[1,0].axhline(0,color='k',linestyle='dashed',linewidth=1)
                mytext(ax[1,0],0.05,0.95,
                       'Median (g-i): %.3f' % np.median(gi),fontsize=FS)
                ax[1,0].set_title('%s' % self.legacy.data.expid[idata],fontsize=FS)
                ax[1,0].set_xlabel('g - i Mag (PS1 + Colorterm)',fontsize=FS)
                ylab= ax[1,0].set_ylabel('(PS1 + CT) - (DECam + ZP)',fontsize=FS)
                ax[1,0].set_xlim(0.4,2.7)
                ax[1,0].set_ylim(-0.5,0.5)
                # 4th panel: r-z vs. g-r
                gr= self.legacy.stars.ps1_g[istar] - self.legacy.stars.ps1_r[istar]
                rz= self.legacy.stars.ps1_r[istar] - self.legacy.stars.ps1_z[istar]
                myscatter(ax[1,1],gr,rz,
                          color='b',m='o',s=50,alpha=0.75) 
                # Arjun's fit
                from scipy.optimize import curve_fit,least_squares
                popt, pcov = curve_fit(stellarlocus, gr,rz,method='lm')
                xp= np.linspace(0, 2, 25)
                ax[1,1].plot(xp, stellarlocus(xp,*popt), 'k--')
                ax[1,1].set_title('%s' % self.legacy.data.expid[idata],fontsize=FS)
                xlab= ax[1,1].set_xlabel('PS1 (g-r) Mag',fontsize=FS)
                ax[1,1].set_ylabel('PS1 (r-z) Mag',fontsize=FS)
                ax[1,1].set_xlim(0,2)
                ax[1,1].set_ylim(0,1.5)
                for row,col in zip(range(2),range(2)):
                    ax[row,row].tick_params(axis='both', labelsize=tickFS)
                savefn="like_arjuns_%s_%s.png" % (base_image_fn,ccdname)
                plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
                plt.close() 
                print("wrote %s" % savefn) 


class Residuals(object):
    """Base class for plotting legacy idl matched zpt and star residuals

    Loads and matches the data, ZptResiduals and StarResiduals do the 
    plotting
    
    Args:
      camera: decam,mosaic,90prime
      savedir: has to write merged zpts file somewhere
      leg_list: list of legacy zpt files
      idl_list: list of idl zpt files
      loadable: False to force merging fits file each time

    Attributes:
      camera: decam,mosaic,90prime
      savedir: has to write merged zpts file somewhere
      leg: LegacyZpt object for legacy table
      idl: ditto for idl table
      loadable: False to force merging fits file each time

    Example:
    leg_fns= glob(os.path.join(os.getenv['CSCRATCH'],
    'dr5_zpts/decam',
    'c4d*-zpt.fits')
    idl_fns= glob(os.path.join(os.getenv['CSCRATCH'],
    'arjundey_Test/AD_exact_skymed',
    'zeropoint*.fits')
    zpt= Residuals(camera='decam',savedir='.',
    leg_list=leg_list,
    idl_list=idl_list,loadable=True)
    zpt.load_data()
    zpt.match() 
    """

    def __init__(self,camera='decam',savedir='./',
                 leg_list=[],
                 idl_list=[],
                 loadable=True):
        self.camera= camera
        self.savedir= savedir
        self.idl= LegacyZpts(zpt_list=idl_list,
                             camera=camera, savedir=savedir, 
                             temptable_name='idl', loadable=loadable)
        self.legacy= LegacyZpts(zpt_list=leg_list,
                                camera=camera,savedir=savedir,
                                temptable_name='legacy',loadable=loadable)

    def load_data(self):
        """Add zeropoints data to LegacyZpts objects"""
        self.idl.load_data()
        self.legacy.load_data()


    def match(self,ra_key='ccdra',dec_key='ccddec'):
        """Cut data to matching ccd centers"""
        m1, m2, d12 = match_radec(self.legacy.data.get(ra_key),self.legacy.data.get(dec_key),
                                  self.idl.data.get(ra_key),self.idl.data.get(dec_key),
                                  1./3600.0,nearest=True)
        self.legacy.data.cut(m1)
        self.idl.data.cut(m2)


class ZptResiduals(Residuals):
    """Matches zeropoint data to idl and plots residuals

    Args:
      camera: decam,mosaic,90prime
      savedir: has to write merged zpts file somewhere
      leg_list: list of legacy zpt files
      idl_list: list of idl zpt files
      loadable: False to merge fits tables each time

    Attributes:
      camera: decam,mosaic,90prime
      savedir: has to write merged zpts file somewhere
      leg: LegacyZpt object for legacy table
      idl: ditto for idl table
      loadable: False to merge fits tables each time

    Example:
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
    zpt.match(ra_key='ccdra',dec_key='ccddec')
    zpt.plot_residuals(doplot='diff') 
    """

    def __init__(self,camera='decam',savedir='./',
                 leg_list=[],
                 idl_list=[],
                 loadable=True):
        super(ZptResiduals, self).__init__(camera,savedir,
                                           leg_list,idl_list,
                                           loadable)
    
    def write_json_expnum2var(self, var,json_fn):
        """writes dict mapping 'expnum' to some variable like 'exptime'"""
        expnums= set(self.legacy.data.expnum)
        d= {str(expnum): float(self.legacy.data.get(var)[
                            self.legacy.data.expnum == expnum
                         ][0]) 
            for expnum in expnums}
        json.dump(d, open(json_fn,'w'))
        print('Wrote %s' % json_fn)

    def convert_legacy(self):
        # Converts to idl names, units
        self.legacy.data= convert_zeropoints_table(self.legacy.data,
                                                   camera=self.camera)


    def get_numeric_keys(self):
        idl_keys= \
             ['exptime','seeing', 'ra', 'dec', 
              'airmass', 'zpt', 'avsky', 
              'fwhm', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 
              'naxis1', 'naxis2', 'ccdnum', 'ccdra', 'ccddec', 
              'ccdzpt', 'ccdphoff', 'ccdphrms', 'ccdskyrms', 'ccdskymag', 
              'ccdskycounts', 'ccdraoff', 'ccddecoff', 'ccdrarms', 'ccddecrms', 'ccdtransp', 
              'ccdnmatch']
        return idl_keys

    #def get_defaultdict_ylim(self,doplot,ylim=None):
    #    ylim_dict=defaultdict(lambda: ylim)
    #    if doplot == 'diff':
    #        ylim_dict['ccdra']= (-1e-8,1e-8)
    #        ylim_dict['ccddec']= ylim_dict['ra']
    #        ylim_dict['ccdzpt']= (-0.005,0.005)
    #    elif doplot == 'div':
    #        ylim_dict['ccdzpt']= (0.995,1.005)
    #        ylim_dict['ccdskycounts']= (0.99,1.01)
    #    else:
    #        pass
    #    return ylim_dict
 
    #def get_defaultdict_xlim(self,doplot,xlim=None):
    #    xlim_dict=defaultdict(lambda: xlim)
    #    if doplot == 'diff':
    #        pass
    #    elif doplot == 'div':
    #        pass
    #    else:
    #        pass
    #    return xlim_dict      


class StarResiduals(Residuals): 
    """Matches star data to idl and plots residuals

    Args:
      camera: decam,mosaic,90prime
      star table: two tables, photom and astrom
      savedir: has to write merged zpts file somewhere
      leg_list: list of legacy zpt files
      idl_list: list of idl zpt files
      loadable: False to merge fits tables each time

    Attributes:
      camera: decam,mosaic,90prime
      star table: two tables, photom and astrom
      savedir: has to write merged zpts file somewhere
      leg: LegacyZpt object for legacy table
      idl: ditto for idl table
      loadable: False to merge fits tables each time

    Example:
    leg_fns= glob(os.path.join(os.getenv['CSCRATCH'],
    'dr5_zpts/decam',
    'c4d*-star.fits')
    idl_fns= glob(os.path.join(os.getenv['CSCRATCH'],
    'arjundey_Test/AD_exact_skymed',
    'matches*.fits')
    zpt= ZptResiduals(camera='decam',savedir='.',
    leg_list=leg_list,
    idl_list=idl_list)
    star.load_data()
    star.convert_legacy()
    star.match(ra_key='ccd_ra',dec_key='ccd_dec')
    star.plot_residuals(doplot='diff')
    """

    def __init__(self,camera='decam',star_table=None,
                 savedir='./',
                 leg_list=[],
                 idl_list=[],
                 loadable=False):
        super(StarResiduals, self).__init__(camera,savedir,
                                            leg_list,idl_list,
                                            loadable) 
        assert(star_table in ['photom','astrom'])
        self.star_table= star_table
    
    def read_json(self, json_fn):
        """return dict"""
        f= open(json_fn, 'r')
        print('Read %s' % json_fn)
        return json.loads(f.read())

    def add_legacy_field(self,name,json_fn=None):
        """adds field 'name' to the legacy table

        Args:
            name: field name to add
            json_fn: if info not in stars table, read from json 
        """
        if json_fn:
            assert(name in ['exptime','gain'])
            d= self.read_json(json_fn)
            new_data= np.zeros(len(self.legacy.data)) - 1
            for expnum in set(self.legacy.data.expnum):
                isExp= self.legacy.data.expnum == expnum
                new_data[isExp]= d[str(expnum)]
        else:
            if name == 'dmagall':
                new_data= (self.legacy.data.ps1_mag -
                            self.legacy.data.apmag)
            else: 
                raise ValueError('name=%s not supported' % name)
        self.legacy.data.set(name, new_data)


    def convert_legacy(self):
        """Converts legay star table to to idl names, units"""
        self.legacy.data= convert_stars_table(self.legacy.data,
                                              camera=self.legacy.camera,
                                              star_table=self.star_table
                                              )

    #def get_numeric_keys(self):
    #    idl_keys= \
    #        ['ccd_x','ccd_y','ccd_ra','ccd_dec',
    #         'ccd_mag','ccd_sky',
    #         'raoff','decoff',
    #         'magoff',
    #         'nmatch',
    #         'gmag','ps1_g','ps1_r','ps1_i','ps1_z']
    #    return idl_keys

    #def get_defaultdict_ylim(self,doplot,ylim=None):
    #    ylim_dict=defaultdict(lambda: ylim)
    #    if doplot == 'diff':
    #        ylim_dict['ccd_ra']= None #(-1e-8,1e-8)
    #        ylim_dict['ccd_dec']= ylim_dict['ra']
    #    elif doplot == 'div':
    #        pass
    #    else:
    #        pass
    #    return ylim_dict
 
    #def get_defaultdict_xlim(self,doplot,xlim=None):
    #    xlim_dict=defaultdict(lambda: xlim)
    #    if doplot == 'diff':
    #        pass
    #    elif doplot == 'div':
    #        pass
    #    else:
    #        pass
    #    return xlim_dict      

    def plot_residuals(self,doplot=None,ms=100,
                       use_keys=None,ylim_dict=None):
        '''two plots of everything numberic between legacy zeropoints and Arjun's
        1) x vs. y-x 
        2) x vs. y/x
        use_keys= list of legacy_keys to use ONLY
        '''
        #raise ValueError
        assert(doplot in ['diff','div'])
        # All keys and any ylims to use
        cols= self.get_numeric_keys()
        #ylim= self.get_defaultdict_ylim(doplot=doplot,ylim=ylim)
        #xlim= self.get_defaultdict_xlim(doplot=doplot,xlim=xlim)
        # Plot
        FS=25
        eFS=FS+5
        tickFS=FS
        bands= np.sort(list(set(self.legacy.data.filter)))
        ccdnames= set(np.char.strip(self.legacy.data.extname))
        for cnt,col in enumerate(cols):
            if use_keys:
                if not col in use_keys:
                    print('skipping col=%s' % col)
                    continue
            # Plot
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            # Color by prim_id = bands + ccdname
            for row,band in zip( range(3), bands ):
                for ccdname,color in zip(ccdnames,['g','r','m','b','k','y']*12):
                    keep= ((self.legacy.data.filter == band) &
                           (np.char.strip(self.legacy.data.extname) == ccdname)
                          )
                    if np.where(keep)[0].size > 0:
                        x= self.idl.data.get( col )[keep]
                        xlabel= 'IDL'
                        if doplot == 'diff':
                            y= self.legacy.data.get(col)[keep] - self.idl.data.get(col)[keep]
                            y_horiz= 0
                            ylabel= 'Legacy - IDL'
                        elif doplot == 'div':
                            y= self.legacy.data.get(col)[keep] / self.idl.data.get(col)[keep]
                            y_horiz= 1
                            ylabel= 'Legacy / IDL'
                        myscatter(ax[row],x,y,color=color,m='o',s=ms,alpha=0.75) 
                        ax[row].axhline(y_horiz,color='k',linestyle='dashed',linewidth=1)
                        #ax[row].text(0.025,0.88,idl_key,\
                        #             va='center',ha='left',transform=ax[cnt].transAxes,fontsize=20)
                        ylab= ax[row].set_ylabel(ylabel,fontsize=FS)
                # Label grz
                ax[row].text(0.9,0.9,band, 
                             transform=ax[row].transAxes,
                             fontsize=FS)
            xlab = ax[2].set_xlabel(xlabel,fontsize=FS)
            supti= ax[0].set_title(col,fontsize=FS)
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                if ylim_dict.get(col,None):
                    ax[row].set_ylim(ylim_dict[col])
                #if xlim[col]:
                #    ax[row].set_xlim(xlim[col])
            savefn=os.path.join(self.savedir,
                                "matchesresiduals_%s_%s.png" % 
                                 (doplot,col))
            plt.savefig(savefn, bbox_extra_artists=[supti,xlab,ylab], 
                        bbox_inches='tight')
            plt.close() 
            print("wrote %s" % savefn)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                            description='Generate a legacypipe-compatible CCDs file \
                                        from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],
                        action='store',required=True)
    parser.add_argument('--data_dir',action='store',
                        default='/home/kaylan/mydata/',required=False)
    parser.add_argument('--idl_dir',action='store',
                        default='/global/cscratch1/sd/kaylanb/arjundey_Test/AD_exact_skymed',
                        required=False)
    args = parser.parse_args()

    from legacyzpts.fetch import fetch_targz
    url_dir= 'http://portal.nersc.gov/project/desi/users/kburleigh/legacyzpts'
    targz_url= os.path.join(url_dir,'idl_legacy_data.tar.gz')
    fetch_targz(targz_url, args.data_dir)

    # default plots
    a=Legacy_vs_IDL(camera=args.camera,
                    leg_dir=args.leg_dir,
                    idl_dir=args.idl_dir)
    # oplot stars on ccd
    run_imshow_stars(os.path.join(args.leg_dir,
                                 'decam/DECam_CP/CP20170326/c4d_170326_233934_oki_z_v1-35-extra.pkl'),
                     arjun_fn=os.path.join(args.idl_dir,
                                           'matches-c4d_170326_233934_oki_z_v1.fits'))
    
    # OLDER:
    # SN
    #sn_not_matched_by_arjun('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-*-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')
    # number star matches
    #number_matches_by_cut('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-*-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')
