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
#from sklearn.neighbors import KernelDensity
from collections import defaultdict
from scipy.stats import sigmaclip

#from PIL import Image, ImageDraw
import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import match_radec
from tractor.sfd import SFDMap
from tractor.brightness import NanoMaggies

from legacyzpts.qa import params
from legacyzpts.qa.paper_plots import LegacyZpts

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
    def __init__(self,camera='decam',
                 leg_dir='/global/cscratch1/sd/kaylanb/kaylan',
                 idl_dir='/global/cscratch1/sd/kaylanb/arjundey_Test'):
        self.camera= camera
        self.leg_dir= leg_dir
        self.idl_dir= idl_dir
        self.fid= params.get_fiducial(camera=self.camera)
        # Zeropoints
        self.idl= LegacyZpts(camera='decam',outdir=self.idl_dir)
        #outdir='/global/cscratch1/sd/kaylanb/zpts_compare_arjun_full60'
        self.legacy= LegacyZpts(camera='decam',outdir=self.leg_dir)
        self.match_zpts()
        self.data_to_idl_units()
        #self.plots()
        #self.plot_nstar()
        # later: 'skymag','skycounts','skyrms','fwhm'
        ms=50
#        for leg_key,idl_key,ylim_arr in zip(['zpt','phrms','rarms','decrms','skymag','skyrms','fwhm'],
#                                            ['ccdzpt','ccdphrms','ccdrarms','ccddecrms','ccdskymag','ccdskyrms','fwhm'],
#                                           [[None],[None],[None],[None],[None],[None]]):
#        for leg_key,idl_key,ylim_arr in zip(['skycounts','skyrms'],
#                                            ['ccdskycounts','ccdskyrms'],
#                                           [[None],[None]]):
#            for cnt,ylim in enumerate(ylim_arr):
#                self.plots_ratios(leg_key=leg_key,idl_key=idl_key,ylim=ylim,prefix='zoom%d' % (cnt+1,),ms=ms)
#
        #for leg_key,idl_key,ylim_arr in zip(['zpt','skymag'],
        #                                    ['ccdzpt','ccdskymag'],
        #                                   [[None],[None]]):
        #    for cnt,ylim in enumerate(ylim_arr):
        #        self.plots_deltas(leg_key=leg_key,idl_key=idl_key,ylim=ylim,prefix='zoom%d' % (cnt+1,),ms=ms)
        ##self.plots_crosshairs(leg_keys=('zpt','phrms'),idl_keys=('ccdzpt','ccdphrms'),ylim=None,prefix='zoom1',ms=ms)
        ##self.plots_crosshairs(leg_keys=('rarms','decrms'),idl_keys=('ccdrarms','ccddecrms'),ylim=None,prefix='zoom1',ms=ms)
        # Stars
        obj= LegacyZpts(camera='decam',outdir=self.idl_dir,what='star_data')
        self.idl.stars= obj.data.copy()
        del obj
        obj= LegacyZpts(camera='decam',outdir=self.leg_dir,what='star_data')
        self.legacy.stars= obj.data.copy()
        del obj
        # Match stars
        self.matched_unmatched_stars()
        # add exptimes,gain,... to star data
        self.add_zpt_data_to_stars()
        # idl ccd_mag -> total electrons in aperature, ccd_sky --> e- in ap from sky
        self.idl.stars.set('obj_Ne_per_sec', self.ccd_mag_to_Ne_per_sec())
        self.idl.stars.set('sky_Ne_per_sec_per_pix', self.ccd_sky_to_Ne_per_sec_per_pix())
        # Same for Legacy
        self.legacy.stars.set('obj_Ne_per_sec', self.legacy.stars.apflux / self.legacy.stars.exptime)
        self.legacy.stars.set('sky_Ne_per_sec_per_pix', self.legacy.stars.apskyflux_perpix / self.legacy.stars.exptime)
        # Sky per sec also useful
        num_pix_in_ap= np.pi * 3.5**2 / self.fid.pixscale**2 #560.639094553 for DECam
        self.legacy.stars.set('sky_Ne_per_sec', num_pix_in_ap * self.legacy.stars.sky_Ne_per_sec_per_pix)
        self.idl.stars.set('sky_Ne_per_sec', num_pix_in_ap * self.idl.stars.sky_Ne_per_sec_per_pix)
        # leg ccd_mag col
        self.legacy.stars.set('ccd_mag', self.apflux_to_idlmag())
        self.legacy.stars.set('ccd_sky', self.apskyflux_to_idlmag())
        # Sky mag in aperture that can compare idl to legacy
        self.skymag_for_idl_legacy()
        # Arjun's measured seeing ('') to fwhm (pixels)
        self.idl.data.set('fwhm_measured',self.idl.data.seeing / self.fid.pixscale)
        print('plotting')
        self.plots_like_arjuns()
        use_keys= ['zpt'] 
        use_keys=None
        self.plot_everything_vs_everything_data(doplot='diff',use_keys=use_keys)
        #self.plot_everything_vs_everything_data(doplot='div',use_keys=use_keys)
        #use_keys= ['ra','dec','obj_Ne_per_sec'] 
        use_keys=None
        #self.plot_everything_vs_everything_stars(doplot='delta_mag',x_eq_ccdmag=True,use_keys=use_keys) 
        #self.plot_everything_vs_everything_stars(doplot='div',x_eq_ccdmag=True,use_keys=use_keys) 
        #self.plot_everything_vs_everything_stars(doplot='div',use_keys=use_keys) 
        self.plot_everything_vs_everything_stars(doplot='diff',use_keys=use_keys) #,ylim=(-1e-8,1e-8))
        raise ValueError
        #self.plot_everything_vs_everything_stars(doplot='div')
        #self.plot_everything_vs_everything_stars(doplot='diff',x_eq_ccdmag=True)
        #self.plot_everything_vs_everything(data_or_stars='data', doplot='diff')
        #self.plot_everything_vs_everything(data_or_stars='data', doplot='div')
        #self.plot_everything_vs_everything(data_or_stars='stars', doplot='diff')
        #self.plot_everything_vs_everything(data_or_stars='stars', doplot='div')
        #self.plot_everything_vs_everything(data_or_stars='stars', doplot='diff',x_eq_mags=True)
        #self.plot_everything_vs_everything(data_or_stars='stars', doplot='div',x_eq_mags=True)
        raise ValueError
        # legacy to idl units
        self.plot_stars_obj_sky_apers_ratios(prefix='',ms=ms)
        self.plot_stars_obj_sky_apers_ratios(prefix='',ms=ms,mags=True)
        self.plot_stars_obj_sky_apflux(prefix='',ms=ms)
        self.plot_stars_obj_sky_apflux(errbars=False,prefix='',ms=ms)
        raise ValueError
        self.plot_stars_obj_sky_chi2(prefix='')
        self.plot_stars_obj_sky_chi2(xlims=1,prefix='xlims')
        #self.plot_stars_obj_sky_apflux(prefix='',ms=ms,sky_per_pix=False)
        self.plot_stars_obj_sky_apflux(prefix='',ms=ms)
        self.plot_stars_obj_sky_apmag(prefix='',ms=ms)
        raise ValueError
        self.idl.stars.set('ccd_rad',np.sqrt(self.idl.stars.ccd_x**2 + self.idl.stars.ccd_y**2))
        self.plot_stars_sky_apers_vs_key(key='ccd_rad',prefix='',ms=ms)
        #self.plot_stars_sky_apers_vs_key(key='ccd_y',prefix='',ms=ms)
        #self.plot_stars_sky_apers_vs_key(key='ccd_y',prefix='',ms=ms)
        raise ValueError
        ms=50
        self.star_plots_dmag_model(prefix='zoom1',ms=ms)
        raise ValueError
        self.star_plots_dmag_model2(ylims=(0.9,1.1),prefix='zoom1',ms=ms)
        raise ValueError
        self.star_plots_dmag(ylims=(0.8,1.4),prefix='zoom1',ms=ms)
        raise ValueError
        self.plots_apskyflux_vs_apskyflux()
        raise ValueError
        self.star_plots_dmag_div_err(ylims=(-5,5),prefix='zoom1',ms=ms)
        self.star_plots_dmag_div_err_sky(prefix='zoom1',ms=ms)
        self.star_plots_dmag_err_bars(prefix='zoom1',ms=ms)
        raise ValueError
        self.star_plots_leg_corrected_w_idl_sky(prefix='zoom1',ms=ms)
        self.star_plots_dmag_dmag(prefix='zoom1',ms=ms)
        self.star_plots_dmag_norm_err(prefix='zoom1',ms=ms)
        self.star_plots_dmag(ylims=(-0.1,0.1),prefix='zoom2',ms=ms)
        #self.star_plots_dmag(ylims=(-0.01,0.01),prefix='zoom3',ms=ms)
        raise ValueError
        self.star_plots()
        self.plot_star_sn()
        self.plot_star_gaiaoffset()


    def get_numeric_keys(self,data_or_stars):
        assert(data_or_stars in ['data','stars'])
        if data_or_stars == 'data':
             legacy_keys= ['airmass','exptime','fwhm','fwhm_cp','gain','image_hdu',
                          'skycounts','skymag','skyrms',
                          'nmatch','nstar','nstarfind',
                          'dec','dec_bore','decoff','decrms',
                          'ra','ra_bore','raoff','rarms',
                          'phoff','phrms','transp',
                          'zpt']
        elif data_or_stars == 'stars':
            legacy_keys= ['obj_Ne_per_sec','sky_Ne_per_sec_per_pix','sky_Ne_per_sec',
                          'ra','dec','x','y']
        # IDL Equivalent
        idl_dict={}
        if data_or_stars == 'data':
            # Add ccd prefix
            for key in ['decoff','decrms','raoff','rarms',
                        'skycounts','skymag','skyrms',
                        'nmatch','nstar','nstarfind',
                        'phoff','phrms','transp',
                        'zpt','zptavg']:
                idl_dict[key]= 'ccd%s' % key
            # Not obvious
            idl_dict['gain']= 'arawgain'
            idl_dict['image_hdu']= 'ccdhdunum'
            idl_dict['dec_bore']= 'dec'
            idl_dict['ra_bore']= 'ra'
            idl_dict['dec']= 'ccddec'
            idl_dict['ra']= 'ccdra'
            idl_dict['fwhm_cp']= 'fwhm'
            idl_dict['fwhm']= 'fwhm_measured'
            # Identical
            for key in legacy_keys:
                if not key in idl_dict.keys():
                    idl_dict[key]= key
        elif data_or_stars == 'stars':
            for key in ['ra','dec','x','y']:
                idl_dict[key] = 'ccd_%s' % key
            # Identical
            for key in legacy_keys:
                if not key in idl_dict.keys():
                    idl_dict[key]= key
        return legacy_keys,idl_dict

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
 
    def get_defaultdict_xlim(self,doplot,xlim=None):
        xlim_dict=defaultdict(lambda: xlim)
        if doplot == 'diff':
            pass
        elif doplot == 'div':
            pass
        else:
            pass
        return xlim_dict      

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
                print "wrote %s" % savefn 


    def plot_everything_vs_everything_data(self,doplot=None,
                                           ms=100,use_keys=None,ylim=None,xlim=None):
        '''two plots of everything numberic between legacy zeropoints and Arjun's
        1) x vs. y-x 
        2) x vs. y/x
        use_keys= list of legacy_keys to use ONLY
        '''
        #raise ValueError
        assert(doplot in ['diff','div'])
        # All keys and any ylims to use
        legacy_keys,idl_dict= self.get_numeric_keys(data_or_stars='data')
        ylim= self.get_defaultdict_ylim(doplot=doplot,ylim=ylim)
        xlim= self.get_defaultdict_xlim(doplot=doplot,xlim=xlim)
        # Plot
        FS=25
        eFS=FS+5
        tickFS=FS
        for cnt,leg_key in enumerate(legacy_keys):
            if use_keys:
                if not leg_key in use_keys:
                    print('skipping legacy_key=%s' % leg_key)
                    continue
            idl_key= idl_dict[leg_key]
            # Plot
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            # Loop over bands, hdus
            for row,band in zip( range(3), self.fid.bands ):
                for hdu,color in zip(set(self.legacy.data.image_hdu),['g','r','m','b','k','y']):
                    keep= (self.legacy.data.filter == band)*\
                          (self.legacy.data.image_hdu == hdu)
                    if np.where(keep)[0].size > 0:
                        x= self.idl.data.get( idl_key )[keep]
                        xlabel= 'IDL'
                        if doplot == 'diff':
                            y= self.legacy.data.get(leg_key)[keep] - self.idl.data.get(idl_key)[keep]
                            y_horiz= 0
                            ylabel= 'Legacy - IDL'
                        elif doplot == 'div':
                            y= self.legacy.data.get(leg_key)[keep] / self.idl.data.get(idl_key)[keep]
                            y_horiz= 1
                            ylabel= 'Legacy / IDL'
                        myscatter(ax[row],x,y,color=color,m='o',s=ms,alpha=0.75) 
                        ax[row].axhline(y_horiz,color='k',linestyle='dashed',linewidth=1)
                        #ax[row].text(0.025,0.88,idl_key,\
                        #             va='center',ha='left',transform=ax[cnt].transAxes,fontsize=20)
                        ylab= ax[row].set_ylabel(ylabel,fontsize=FS)
            xlab = ax[row].set_xlabel(xlabel,fontsize=FS)
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                if ylim[leg_key]:
                    ax[row].set_ylim(ylim[leg_key])
                if xlim[leg_key]:
                    ax[row].set_xlim(xlim[leg_key])
            savefn="everything_data_%s_%s_%s.png" % (self.camera,doplot,leg_key)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 


    def plot_everything_vs_everything_stars(self,doplot=None,x_eq_ccdmag=False,
                                           ms=100,use_keys=None,xlim=None,ylim=None,
                                           prefix=''):
        '''two plots of everything numberic between legacy zeropoints and Arjun's
        1) x vs. y-x 
        2) x vs. y/x
        use_keys= list of legacy_keys to use ONLY
        x_eq_mags -- only for data_or_stars="stars", plots y-axis vs. x=ccd_mag
        '''
        #raise ValueError
        assert(doplot in ['diff','div','delta_mag'])
        # All keys and any ylims to use
        legacy_keys,idl_dict= self.get_numeric_keys(data_or_stars='stars')
        ylim= self.get_defaultdict_ylim(doplot=doplot,ylim=ylim)
        xlim= self.get_defaultdict_xlim(doplot=doplot,xlim=xlim)
        # Plot
        FS=25
        eFS=FS+5
        tickFS=FS
        for cnt,leg_key in enumerate(legacy_keys):
            if use_keys:
                if not leg_key in use_keys:
                    print('skipping legacy_key=%s' % leg_key)
                    continue
            if doplot == 'delta_mag':
                if not leg_key in ['obj_Ne_per_sec']:
                    print('skipping legacy_key=%s' % leg_key)
                    continue
            idl_key= idl_dict[leg_key]
            # Plot
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            # Loop over bands, hdus
            for row,band in zip( range(3), self.fid.bands ):
                for hdu,color in zip(set(self.legacy.stars.image_hdu),['g','r','m','b','k','y']):
                    keep= (self.legacy.stars.filter == band)*\
                          (self.legacy.stars.image_hdu == hdu)
                    if np.where(keep)[0].size > 0:
                        if x_eq_ccdmag:
                            x= self.idl.stars.ccd_mag[keep]
                            xlabel= 'ccd_mag (IDL)'
                        else:
                            x= self.idl.stars.get( idl_key )[keep]
                            xlabel= 'IDL'
                        if doplot == 'diff':
                            y= self.legacy.stars.get(leg_key)[keep] - self.idl.stars.get(idl_key)[keep]
                            y_horiz= 0
                            ylabel= 'Legacy - IDL'
                        elif doplot == 'div':
                            y= self.legacy.stars.get(leg_key)[keep] / self.idl.stars.get(idl_key)[keep]
                            y_horiz= 1
                            ylabel= 'Legacy / IDL'
                        elif doplot == 'delta_mag':
                            y= -2.5*np.log10(self.legacy.stars.get(leg_key)[keep] / self.idl.stars.get(idl_key)[keep])
                            y_horiz= 0
                            ylabel= 'Legacy - IDL'
                            prefix= 'DeltaMag'
                        myscatter(ax[row],x,y,color=color,m='o',s=ms,alpha=0.75) 
                        print('leg_key=%s, median y=' % leg_key,np.median(y)) 
                        ax[row].axhline(y_horiz,color='k',linestyle='dashed',linewidth=1)
                        #ax[row].text(0.025,0.88,idl_key,\
                        #             va='center',ha='left',transform=ax[cnt].transAxes,fontsize=20)
                        ylab= ax[row].set_ylabel(ylabel,fontsize=FS)
            xlab = ax[row].set_xlabel(xlabel,fontsize=FS)
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                if ylim[leg_key]:
                    ax[row].set_ylim(ylim[leg_key])
                try:
                    if xlim[leg_key]:
                        ax[row].set_xlim(xlim[leg_key])
                except:
                    raise ValueError
            savefn="everything_stars_%s_%s_xeqccdmag%s_%s%s.png" % (self.camera,doplot,x_eq_ccdmag,leg_key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 



    def plot_stars_obj_sky_apflux(self,prefix='',xlims=None,ylims=None,ms=50,
                                  errbars=True,annotate=False):
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        num_pix_in_ap= np.pi * 3.5**2 / self.fid.pixscale**2 #560.639094553 for DECam
        for row,band in zip( range(3), self.fid.bands ):
            for hdu,color in zip(set(self.legacy.stars.image_hdu),['g','r','m','b','k','y']):
                keep= (self.legacy.stars.filter == band)*\
                      (self.legacy.stars.image_hdu == hdu)
                if np.where(keep)[0].size > 0:
                    x= self.legacy.stars.sky_Ne_per_sec[keep] -\
                       self.idl.stars.sky_Ne_per_sec[keep] 
                    xerr= np.sqrt(self.legacy.stars.sky_Ne_per_sec[keep] +\
                                  self.idl.stars.sky_Ne_per_sec[keep])
                    y= self.legacy.stars.obj_Ne_per_sec[keep] - \
                       self.idl.stars.obj_Ne_per_sec[keep] 
                    yerr= np.sqrt(self.legacy.stars.obj_Ne_per_sec[keep] + \
                                  self.legacy.stars.sky_Ne_per_sec[keep] +\
                                  self.idl.stars.obj_Ne_per_sec[keep] +\
                                  self.idl.stars.sky_Ne_per_sec[keep])
                    xlab='sky_Ne_per_sec_per_pix (legacy - idl)'
                    ylab='obj_Ne_per_sec (legacy - idl)'
                    if errbars:
                        myerrorbar(ax[row],x,y, yerr=yerr,xerr=xerr,color=color,m='o',s=10.,alpha=0.75)
                    else:
                        myscatter(ax[row],x,y,color=color,m='o',s=ms,alpha=0.75)
                    # Annotate with SN
                    if annotate:
                        SN= self.legacy.stars.obj_Ne_per_sec[keep] / \
                            np.sqrt(self.legacy.stars.obj_Ne_per_sec[keep] + \
                                        self.legacy.stars.sky_Ne_per_sec_per_pix[keep]*\
                                        num_pix_in_ap)
                        myannot(ax[row],x,y, np.around(SN,decimals=0).astype(np.string_), fontsize=15)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    ax[row].axvline(0,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel(ylab,fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
        xlab=ax[2].set_xlabel(xlab,fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='%s_obj_sky_apflux_errbars%s_%s.png' % (self.camera,errbars,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def plot_stars_obj_sky_apmag(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        for row,band in zip( range(3), self.fid.bands ):
            for hdu,color in zip(set(self.legacy.stars.image_hdu),['g','r','m','b','k','y']):
                keep= (self.legacy.stars.filter == band)*\
                      (self.legacy.stars.image_hdu == hdu)
                if np.where(keep)[0].size > 0:
                    x= -2.5*np.log10(self.legacy.stars.sky_Ne_per_sec_per_pix[keep] /\
                                     self.idl.stars.sky_Ne_per_sec_per_pix[keep])
                    y= -2.5*np.log10(self.legacy.stars.obj_Ne_per_sec[keep] / \
                                     self.idl.stars.obj_Ne_per_sec[keep] )
                    xlab='mag sky_Ne_per_sec_per_pix [-2.5log10(legacy idl)]'
                    ylab='mag obj_Ne_per_sec [-2.5log10(legacy - idl)]'
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    ax[row].axvline(0,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel(ylab,fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
        xlab=ax[2].set_xlabel(xlab,fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='%s_obj_sky_apmag_%s.png' % (self.camera,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def plot_stars_obj_sky_chi2(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        for key in ['sky_Ne_per_sec','obj_Ne_per_sec']:
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band in zip( range(3), self.fid.bands ):
                for hdu,color in zip(set(self.legacy.stars.image_hdu),['g','r','m','b','k','y']):
                    keep= (self.legacy.stars.filter == band)*\
                          (self.legacy.stars.image_hdu == hdu)
                    if np.where(keep)[0].size > 0:
                        dflux= self.legacy.stars.get(key)[keep] -\
                               self.idl.stars.get(key)[keep]
                        if key == 'sky_Ne_per_sec':
                            err= np.sqrt(self.legacy.stars.get(key)[keep] + \
                                         self.idl.stars.get(key)[keep])
                        if key == 'obj_Ne_per_sec':
                            err= np.sqrt(self.legacy.stars.get(key)[keep] + \
                                         self.legacy.stars.sky_Ne_per_sec[keep] + \
                                         self.idl.stars.get(key)[keep] +\
                                         self.idl.stars.sky_Ne_per_sec[keep])
                        chi2= dflux / err
                        h,bins= myhist_step(ax[row],chi2,bins=20,color=color,normed=True,lw=2,return_vals=True)
                        sanity= np.sum(h*(bins[1:]-bins[:-1]))
                        print('sanity = 1? = %f' % sanity)
                        ylab='PDF'
                        xlab='%s (Legacy - IDL) / sqrt(Var)' % key
                        # Unit gaussian N(0,1)
                        from scipy.stats import norm as scipy_norm
                        G= scipy_norm(0,1)
                        xvals= np.linspace(chi2.min(),chi2.max())
                        ax[row].plot(xvals,G.pdf(xvals),'k--',lw=2)
                        # lines through 0
                        ax[row].axvline(0,color='k',linestyle='dashed',linewidth=1)
                        # Label
                        ylab=ax[row].set_ylabel(ylab,fontsize=FS)
                        if xlims:
                            if key == 'sky_Ne_per_sec':
                                xlims=(-0.2,1.2)
                            if key == 'obj_Ne_per_sec':
                                xlims=(-1.2,0.2)
                            ax[row].set_xlim(xlims)
                        if ylims:
                            ax[row].set_ylim(ylims)
            xlab=ax[2].set_xlabel(xlab,fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='%s_obj_sky_chi2_%s_%s.png' % (self.camera,key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 



    def plot_stars_obj_sky_apers_ratios(self,prefix='',xlims=None,ylims=None,ms=50,mags=False):
        FS=25
        eFS=FS+5
        tickFS=FS
        for key in ['obj_Ne_per_sec','sky_Ne_per_sec']:
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band in zip( range(3), self.fid.bands ):
                for hdu,color in zip(set(self.legacy.stars.image_hdu),['g','r','m','b','k','y']):
                    print('hdu=',hdu,'color=%s' % color)
                    keep= (self.legacy.stars.filter == band)*\
                          (self.legacy.stars.image_hdu == hdu)
                    if np.where(keep)[0].size > 0:
                        x= self.idl.stars.ccd_mag[keep] 
                        y= self.legacy.stars.get(key)[keep] / self.idl.stars.get(key)[keep] 
                        ylab= '(legacy/idl)'
                        y_goal= 1.
                        if mags:
                            y= -2.5*np.log10(y)
                            ylab= '-2.5 log10(legacy/idl)'
                            y_goal= 0.
                        myscatter(ax[row],x,y,
                                  color=color,m='o',s=ms,alpha=0.75)
                        # HDU factors
                        #ref_gain=4.3
                        #for gain in [4.51,4.47,4.4,4.3]:
                        #    ax[row].axhline(gain/ref_gain,color='k',linestyle='dashed',linewidth=1)
                        ax[row].axhline(y_goal,color='k',linestyle='dashed',linewidth=1)
                        # Label
                        ylab=ax[row].set_ylabel('%s %s' % (key,ylab),fontsize=FS)
                        if xlims:
                            ax[row].set_xlim(xlims)
                        if ylims:
                            ax[row].set_ylim(ylims)
            ax[0].set_ylim(-0.4,0.6)
            ax[1].set_ylim(-0.15,0.3)
            ax[2].set_ylim(-0.1,0.35)
            xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_obj_sky_apers_ratio_%s_mags%s%s.png' % (self.camera,key,mags,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def plot_stars_sky_apers_vs_key(self,key='x',prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        for row,band in zip( range(3), self.fid.bands ):
            for hdu,color in zip(set(self.legacy.stars.image_hdu),['g','r','m','b','k','y']):
                keep= (self.legacy.stars.filter == band)*\
                      (self.legacy.stars.image_hdu == hdu)
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.get(key)[keep] 
                    y= self.legacy.stars.sky_Ne_per_sec_per_pix[keep] - self.idl.stars.sky_Ne_per_sec_per_pix[keep] 
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(1,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('sky_Ne_per_sec_per_pix (legacy - idl)',fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
        xlab=ax[2].set_xlabel("%s (idl)" % key,fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_sky_apers_vs_%s%s.png' % (self.camera,key,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 




    def add_zpt_data_to_stars(self):
        '''Assuming self.legacy.stars and self.idl.stars are matched samples'''
        assert(len(self.legacy.stars) == len(self.idl.stars))
        # Have all the info in legacy, legacy zpt -> legacy stars
        exptime= np.zeros(len(self.legacy.stars))
        gain= np.zeros(len(self.legacy.stars))
        # loop through
        expnums= set(self.legacy.data.expnum)
        hdus= set(self.legacy.data.image_hdu)
        for expnum in set(self.legacy.stars.expnum):
            for hdu in set(self.legacy.stars.image_hdu):
                assert(expnum in expnums)
                assert(hdu in hdus)
                indata= (self.legacy.data.expnum == expnum)*(self.legacy.data.image_hdu == hdu)
                instars= (self.legacy.stars.expnum == expnum)*(self.legacy.stars.image_hdu == hdu)
                # Store
                exptime[ instars ]= self.legacy.data.exptime[ indata ][0]
                gain[ instars ]= self.legacy.data.gain[ indata ][0]
        # Add to table
        self.legacy.stars.set('exptime',exptime)
        self.legacy.stars.set('gain',gain)
        # Now legacy stars -> idl_stars
        self.idl.stars.set('exptime',exptime)
        self.idl.stars.set('gain',gain)
        # Add filter as well
        self.idl.stars.set('filter',self.legacy.stars.filter)
        #filter= np.zeros(len(self.idl.stars)).astype(str)
        #filter[ instars ]= self.idl.data.filter[ indata ][0]
        #self.idl.stars.set('filter',filter)

    def ccd_mag_to_Ne_per_sec(self):
        '''IDL aper.pro returns "mags" as -2.5log10(Ne) + 25
        In arjuns code, Ne is in units of ADU which he converts to e- and thereofer AB mags with
        ccd_mag= "mags" - 25 + 2.5log10(exptime) + zp0
            where zp0 = 26.610,26.818,26.484 (g,r,z) - 2.5log10(gain)
            and gain is ARAWGAIN from the header
        '''
        data= np.zeros(len(self.idl.stars))
        for band in self.fid.bands:
            keep= self.idl.stars.filter == band
            #data[keep]= self.idl.stars.exptime[keep] * \
            #            10**(-1/2.5 * (self.idl.stars.ccd_mag[keep] - self.fid.zp0[band]))
            data[keep]= 10**(-1/2.5 * (self.idl.stars.ccd_mag[keep] - self.fid.zp0[band]))
        return data

    def ccd_sky_to_Ne_per_sec_per_pix(self):
        '''IDL aper.pro returns sky in as the Mode of Counts in the 7-10 arcsec sky aperture
        for arjuns code Counts is ADU so 
        sky Ne / pix / sec (e/pix/sec)= ccd_sky (ADU) * gain / exptime'''
        #num_pix_in_ap= np.pi * 7**2 / self.fid.pixscale**2 #560.639094553 for DECam
        #data= np.zeros(len(self.idl.stars))
        #for band in self.fid.bands:
        #    keep= self.idl.stars.filter == band
        #    data[keep]= self.idl.stars.ccd_sky[keep] * npix_7arcsec_ap
        #return data
        #return num_pix_in_ap * self.idl.stars.ccd_sky # * self.idl.stars.gain 
        return self.idl.stars.ccd_sky * self.idl.stars.gain / self.idl.stars.exptime


    def skymag_for_idl_legacy(self):
        num_pix_in_ap= np.pi * 7**2 / self.fid.pixscale**2
        # IDL
        idl_Ne= self.idl.stars.ccd_sky * num_pix_in_ap # idl ccd_sky: ADU / pixel * gain * N pixels
        # units e/pix?? * self.idl.stars.gain
        idl_skymag= np.zeros(len(self.idl.stars))
        for band in self.fid.bands:
            keep= self.idl.stars.filter == band
            idl_skymag[keep]= -2.5 * np.log10(idl_Ne[keep] / self.idl.stars.exptime[keep]) +  \
                              self.fid.zp0[band]
        self.idl.stars.set('skymag_inap', idl_skymag)
        # Legacy
        leg_Ne= self.legacy.stars.apskyflux_perpix * num_pix_in_ap # leg apsky: e / pixel * N pixels
        leg_skymag= np.zeros(len(self.legacy.stars))
        for band in self.fid.bands:
            keep= self.legacy.stars.filter == band
            leg_skymag[keep]= -2.5 * np.log10(leg_Ne[keep] / self.legacy.stars.exptime[keep]) +  \
                              self.fid.zp0[band]
        self.legacy.stars.set('skymag_inap', leg_skymag)


    def apflux_to_idlmag(self):
        '''leg apflux to idl ccd_mag'''
        data= np.zeros(len(self.legacy.stars))
        for band in self.fid.bands:
            keep= self.legacy.stars.filter == band
            data[keep]= -2.5 * np.log10(self.legacy.stars.apflux[keep] / self.legacy.stars.exptime[keep]) +  \
                        self.fid.zp0[band]
        return data

    def apskyflux_to_idlmag(self):
        '''leg apflux to idl ccd_mag'''
        data= np.zeros(len(self.legacy.stars))
        for band in self.fid.bands:
            keep= self.legacy.stars.filter == band
            data[keep]= -2.5 * np.log10(self.legacy.stars.apskyflux[keep] / self.legacy.stars.exptime[keep]) +  \
                        self.fid.zp0[band]
        return data

    def matched_unmatched_stars(self):
        ''' Matched: legacy.stars,idl.stars
        Unmatched: .stars_unm
        '''
        self.legacy.stars_unm= self.legacy.stars.copy()
        self.idl.stars_unm= self.idl.stars.copy()
        # matched
        m1, m2, d12 = match_radec(self.legacy.stars.ra,self.legacy.stars.dec,
                                  self.idl.stars.ccd_ra,self.idl.stars.ccd_dec, 
                                  1./3600.0,nearest=True)
        self.legacy.stars.cut(m1)
        self.idl.stars.cut(m2)
        # Not matched
        unm1= np.delete(np.arange(len(self.legacy.stars)),m1)
        unm2= np.delete(np.arange(len(self.idl.stars)),m2)
        self.legacy.stars_unm.cut(unm1)
        self.idl.stars_unm.cut(unm2)


    def match_zpts(self):
        ## Save time
        #if os.path.exists(self.save_fn):
        #    self.load()
        #    # Cuts afer looking at idl vs. legacy plots
        #    if self.camera == 'mosaic':
        #        self.tab.cut( self.tab.anot_transp_med < 2)
        #        self.tab.cut( self.tab.anot_transp_med < 30)
        #else:
        m1, m2, d12 = match_radec(self.legacy.data.ra,self.legacy.data.dec,
                                  self.idl.data.ccdra,self.idl.data.ccddec, 
                                  60./3600.0,nearest=True)
        self.legacy.data= self.legacy.data[m1] 
        self.idl.data= self.idl.data[m2] 
        #expnums= set(self.idl.data.expnum)
        #d= defaultdict(list)
        #for i,expnum in enumerate(expnums):
        #    ccdnames= set(self.
        #    for j,ccdname in enumerate(
        #    if (i+1) % 100 == 0:
        #        print('match_expnum_ccdname: %d/%d' % (i+1,len(expnums)))
        #    keep= self.legacy.data.expnum == expnum
        #    if len(legacy.data[keep]) > 0:
        #        band= legacy.data.filter[keep][0]
        #        exptime= legacy.data.exptime[keep][0]
        #        #assert(band == np.string_.strip(db.data.band[i]))
        #        idl_keep= self.idl.data.expnum == expnum
        #        assert(band == self.idl.data.filter[idl_keep][0])
        #        assert(int(exptime) == int(self.idl.data.exptime[idl_keep][0]))
        #        d['expnum'].append( expnum )
        #        d['filter'].append( band )
        #        d['exptime'].append( exptime )
        #        # Me
        #        for key in ['skymag','zpt','transp','fwhm','fwhm2']:
        #            d['legacy_'+key+'_med'].append( np.median(legacy.data.get(key)[keep]) ) 
        #            d['legacy_'+key+'_std'].append( np.std(legacy.data.get(key)[keep]) ) 
        #        # db
        #        for key,dbkey in zip(['skymag','zpt','transp','fwhm','fwhm2'],
        #                             ['ccdskymag','ccdzpt','ccdtransp','fwhm','fwhm']):
        #            d['anot_'+key+'_med'].append( np.median(anot.get(dbkey)[anot_keep]) ) 
        #            d['anot_'+key+'_std'].append( np.std(anot.get(dbkey)[anot_keep]) ) 
        #    # Repackage
        #    self.tab= fits_table()
        #    for key in d.keys():
        #        self.tab.set(key, np.array(d[key]) )
        #    print('compare_me_db fits_table: %d' % len(self.tab))
        #    # my tneeded exposure time can be set to tmin, 
        #    # but db tneed can be larger tmin by a lot
        #    #keep= np.ones(len(self.tab),bool)
        #    #for band in self.fid.bands:
        #    #    rem= (self.tab.filter == band) * \
        #    #         (self.tab.me_tneed_med.astype(int) == int(self.fid.tmin[band])) * \
        #    #         (self.tab.db_tneed.astype(int) > 10 + int(self.fid.tmin[band]))
        #    #    keep[rem]= False
        #    #self.tab.cut(keep)
        #    #print('compare_me_db after remove tneed=min, db tneed >> tmin: %d' % len(self.tab))
        #    # Save
        #    self.save()
    
    def data_to_idl_units(self):
        for key in ['skycounts','skyrms']:
            self.legacy.data.set(key, self.legacy.data.get(key) * self.legacy.data.exptime / self.legacy.data.gain)
        for key in ['zpt']:
            self.legacy.data.set(key, self.legacy.data.get(key) - 2.5*np.log10(self.legacy.data.gain))

    def save(self):
        self.tab.writeto(self.save_fn)
        print('Wrote %s' % self.save_fn)
    
    def load(self):
        self.tab= fits_table(self.save_fn)
        print('Loaded %s' % self.save_fn)

    def plots(self,prefix=''):
        FS=25
        eFS=FS+5
        tickFS=FS
        xlims,ylims= None,None
        for leg_key,idl_key in zip(['skymag','skycounts','skyrms',
                                    'zpt','transp','fwhm','phoff','rarms',
                                    'decrms'],
                                   ['ccdskymag','ccdskycounts','ccdskyrms',
                                    'ccdzpt','ccdtransp','fwhm','ccdphoff','ccdrarms',
                                    'ccddecrms']): 
            fig,ax= plt.subplots(3,1,figsize=(12,15))
            plt.subplots_adjust(hspace=0.2,wspace=0)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.data.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.legacy.data.get(leg_key)[keep]
                    y= self.idl.data.get(idl_key)[keep]
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=10.,alpha=0.75, 
                              label='%d' % len(x))
                    # fit line
                    try:
                        popt= np.polyfit(x,y, 1)
                    except:
                        raise ValueError
                    x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s (idl)' % idl_key,fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims[band])
                    if ylims:
                        ax[row].set_ylim(ylims[band])
            xlab=ax[2].set_xlabel("%s (legacy)" % leg_key,fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                ax[row].legend(loc='upper left',fontsize=FS)
            savefn='legacy_vs_idl_%s_%s.png' % (self.camera,leg_key)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def plots_deltas(self,leg_key='zpt',idl_key='ccdzpt',ylim=None,prefix='',ms=50):
        '''zpt - zpt vs. zpt, skymag - skymag vs skymag etc
        '''
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(12,15))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
            keep= self.legacy.data.filter == band
            if np.where(keep)[0].size > 0:
                x= self.legacy.data.get(leg_key)[keep]
                y= self.idl.data.get(idl_key)[keep]
                myscatter(ax[row],y,x - y,
                          color=color,m='o',s=ms,alpha=0.75)
                # Label
                ylab=ax[row].set_ylabel('%s %s (legacy - idl)' % (band,idl_key),fontsize=FS)
                if ylim:
                    ax[row].set_ylim(ylim)
        xlab=ax[2].set_xlabel("%s (idl)" % leg_key,fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            ax[row].legend(loc='upper left',fontsize=FS)
            # lines through 1
            ax[row].axhline(1,color='k',linestyle='dashed',linewidth=1)
        savefn='legacy_vs_idl_deltas_%s_%s%s.png' % (self.camera,leg_key,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 



    def plots_ratios(self,leg_key='zpt',idl_key='ccdzpt',ylim=None,prefix='',ms=50):
        '''zpt/zpt vs. zpt, rarms/rarms vs rarms etc
        Useful in tadem with plot_crosshairs 
        phrms/phrms vs. zpt/zpt
        decrms/decrms vs. rarms/rarms'''
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(12,15))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
            keep= self.legacy.data.filter == band
            if np.where(keep)[0].size > 0:
                x= self.legacy.data.get(leg_key)[keep]
                y= self.idl.data.get(idl_key)[keep]
                myscatter(ax[row],y,x/y,
                          color=color,m='o',s=ms,alpha=0.75)
                # Label
                ylab=ax[row].set_ylabel('%s %s (legacy/idl)' % (band,idl_key),fontsize=FS)
                if ylim:
                    ax[row].set_ylim(ylim)
        xlab=ax[2].set_xlabel("%s (idl)" % leg_key,fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            ax[row].legend(loc='upper left',fontsize=FS)
            # lines through 1
            ax[row].axhline(1,color='k',linestyle='dashed',linewidth=1)
        savefn='legacy_vs_idl_ratios_%s_%s%s.png' % (self.camera,leg_key,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def plots_crosshairs(self,leg_keys=('zpt','phrms'),idl_keys=('ccdzpt','ccdphrms'),ylim=None,prefix='',ms=50):
        '''crosshair plots of 
        phrms/phrms vs. zpt/zpt
        decrms/decrms vs. rarms/rarms'''
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(12,15))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
            keep= self.legacy.data.filter == band
            if np.where(keep)[0].size > 0:
                x= self.legacy.data.get(leg_keys[0])[keep] / self.idl.data.get(idl_keys[0])[keep]
                y= self.legacy.data.get(leg_keys[1])[keep] / self.idl.data.get(idl_keys[1])[keep]
                myscatter(ax[row],x,y,
                          color=color,m='o',s=ms,alpha=0.75)
                # Label
                ylab=ax[row].set_ylabel('%s %s (legacy/idl)' % (band,idl_keys[1]),fontsize=FS)
                if ylim:
                    ax[row].set_ylim(ylim)
        xlab=ax[2].set_xlabel("%s (legacy/idl)" % idl_keys[0],fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            ax[row].legend(loc='upper left',fontsize=FS)
            # lines through 1
            ax[row].axhline(1,color='k',linestyle='dashed',linewidth=1)
            ax[row].axvline(1,color='k',linestyle='dashed',linewidth=1)
        savefn='legacy_vs_idl_crosshairs_%s_%s_%s%s.png' % (self.camera,leg_keys[0],leg_keys[1],prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def plot_nstar(self,prefix=''):
        FS=25
        eFS=FS+5
        tickFS=FS
        xlims,ylims= None,None
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.5,wspace=0)
        for row,band in zip( range(3), self.fid.bands):
            for leg_key,idl_key,color in zip(['nstarfind','nstar','nmatch'],
                                             ['ccdnstarfind','ccdnstar','ccdnmatch'],
                                             ['g','r','m']): 
                keep= self.legacy.data.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.legacy.data.get(leg_key)[keep]
                    y= self.idl.data.get(idl_key)[keep]
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=20.,alpha=0.75, 
                              label='%s' % leg_key)
                    # fit line
                    popt= np.polyfit(x,y, 1)
                    x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    ax[row].plot(x,popt[0]*x + popt[1],c=color,ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
            ylab=ax[row].set_ylabel('%s N stars (idl)' % band,fontsize=FS)
        xlab=ax[2].set_xlabel("N stars (legacy)",fontsize=FS) #0.45'' galaxy
        for row in range(3)[::-1]:
            ax[row].tick_params(axis='both', labelsize=tickFS)
            leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_%s.png' % (self.camera,'nstar')
        plt.savefig(savefn,bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def star_plots(self,prefix=''):
        FS=25
        eFS=FS+5
        tickFS=FS
        xlims,ylims= None,None
        for leg_key,idl_key in zip(['ccd_mag','apflux','apskyflux'],
                                   ['ccd_mag','apflux','apskyflux']):
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.stars.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.legacy.stars.get(leg_key)[keep]
                    y= self.idl.stars.get(idl_key)[keep]
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=10.,alpha=0.75, 
                              label='%d' % len(x))
                    # fit line
                    try:
                        popt= np.polyfit(x,y, 1)
                    except:
                        raise ValueError
                    x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s (idl)' % idl_key,fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims[band])
                    if ylims:
                        ax[row].set_ylim(ylims[band])
            xlab=ax[2].set_xlabel("%s (legacy)" % leg_key,fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_%s.png' % (self.camera,leg_key)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 


    def star_plots_dmag(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        for leg_key,idl_key in zip(['ccd_mag','skymag_inap','apflux','apskyflux'],
                                   ['ccd_mag','skymag_inap','apflux','apskyflux']):
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.stars.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.ccd_mag[keep]
                    y= self.legacy.stars.get(leg_key)[keep] / self.idl.stars.get(idl_key)[keep] 
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s %s (legacy / idl)' % (band,idl_key),fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
            xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_dmag_%s_%s.png' % (self.camera,leg_key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def star_plots_dmag_model(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
            keep= self.legacy.stars.filter == band
            if np.where(keep)[0].size > 0:
                x= self.idl.stars.ccd_mag[keep]
                y= self.legacy.stars.apflux[keep] - self.idl.stars.apflux[keep] 
                stddev= np.sqrt(self.legacy.stars.apflux[keep] +\
                                self.idl.stars.apflux[keep] +\
                                self.legacy.stars.apskyflux[keep] +\
                                self.idl.stars.apskyflux[keep])
                myscatter(ax[row],x,y / stddev,
                          color=color,m='o',s=ms,alpha=0.75)
                # lines through 0
                #ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                # Model
                xarr_dic=dict(g=np.linspace(16,23,num=20),
                             r=np.linspace(15,21,num=20),
                             z=np.linspace(15,20,num=20))
                xarr= xarr_dic[band]
                dmag= 0.1
                Oa,Sa= [],[]
                for xa in xarr:
                    samp= (keep) * (self.idl.stars.ccd_mag > xa-dmag)*(self.idl.stars.ccd_mag < xa+dmag)
                    Oa.append( self.idl.stars.apflux[samp].mean() )
                    Sa.append( self.idl.stars.apskyflux[samp].mean() )
                Oa= np.array(Oa)
                Sa= np.array(Sa)
                print('Oa= ',Oa)
                print('Sa= ',Sa)
                for f,ls in zip([1.01,1.05,1.1],['dotted','solid','dashed']):
                    y_mod= -Sa * (f-1) / np.sqrt(2*Oa + 2*Sa)
                    print('y_mod=',y_mod)
                    ax[row].plot(xarr, y_mod,c='k',ls=ls,lw=2,label='f = %.2f' % f)
                # more points
                #Oa= self.idl.stars.apflux[keep]
                #Sa= self.idl.stars.apskyflux[keep]
                #f=1.2
                #y_mod= -Sa * (f-1) / np.sqrt(2*Oa + 2*Sa)
                #print('y_mod=',y_mod)
                #myscatter(ax[row],x,y_mod,
                #          color='k',m='o',s=ms,alpha=0.75)
                # fit line
                #try:
                #    popt= np.polyfit(x,y, 1)
                #except:
                #    raise ValueError
                #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                # slope 1
                #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                # Label
                ylab=ax[row].set_ylabel('%s %s (legacy - idl)' % (band,'apflux'),fontsize=FS)
                if xlims:
                    ax[row].set_xlim(xlims)
                if ylims:
                    ax[row].set_ylim(ylims)
                ax[row].set_ylim(-20,1)
                #ax[row].set_yscale('log')
                #ax[row].set_xscale('log')
        leg=ax[0].legend(loc=(0,1.01),ncol=3)
        xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_dmag_model_%s_%s.png' % (self.camera,'apskyflux',prefix)
        plt.savefig(savefn, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 




    def star_plots_dmag_model2(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        for leg_key,idl_key in zip(['apflux'],
                                   ['apflux']):
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.stars.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.ccd_mag[keep]
                    y= self.legacy.stars.get(leg_key)[keep] / self.idl.stars.get(idl_key)[keep] 
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    # Model
                    xarr= np.linspace(16,20,num=10)
                    dmag= 0.1
                    Oa,Sa= [],[]
                    for xa in xarr:
                        samp= (self.idl.stars.ccd_mag[keep] > xa-dmag)*(self.idl.stars.ccd_mag[keep] < xa+dmag)
                        Oa.append( self.idl.stars.apflux[samp].mean() )
                        Sa.append( self.idl.stars.apskyflux[samp].mean() )
                    Oa= np.array(Oa)
                    Sa= np.array(Sa)
                    #Oa= self.idl.stars.apflux[keep]
                    #Sa= self.idl.stars.apskyflux[keep]
                    for f,ls in zip([1.,1.2],['dotted','dashed']):
                        y_mod= (Oa + (1-f)*Sa) / Oa
                        ax[row].plot(xarr, y_mod,c='k',ls=ls,lw=2)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s %s (legacy / idl)' % (band,idl_key),fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
                    ax[row].set_yscale('log')
                    #ax[row].set_xscale('log')
            xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_dmag_model2_%s_%s.png' % (self.camera,leg_key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 



    def star_plots_leg_corrected_w_idl_sky(self,prefix='',xlims=None,ylims=None,ms=50):
        ''''''
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
            keep= self.legacy.stars.filter == band
            if np.where(keep)[0].size > 0:
                raise ValueError
                apskyflux_diff= self.legacy.stars.apskyflux[keep] - self.idl.stars.apskyflux[keep] 
                Ne_leg= self.legacy.stars.apflux[keep] + apskyflux_diff
                Ne_idl= self.idl.stars.apflux[keep]
                x= apskyflux_diff
                y= Ne_leg / Ne_idl
                myscatter(ax[row],x,y,
                          color=color,m='o',s=ms,alpha=0.75)
                # lines through 1
                ax[row].axhline(1,color='k',linestyle='dashed',linewidth=1)
                # fit line
                #try:
                #    popt= np.polyfit(x,y, 1)
                #except:
                #    raise ValueError
                #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                # slope 1
                #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                # Label
                ylab=ax[row].set_ylabel('%s f_leg\'/f_idl' % (band,),fontsize=FS)
                if xlims:
                    ax[row].set_xlim(xlims)
                if ylims:
                    ax[row].set_ylim(ylims)
        xlab=ax[2].set_xlabel("sky_leg - sky_idl",fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_fleg_fidl_%s.png' % (self.camera,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 



    def star_plots_dmag_dmag(self,prefix='',xlims=None,ylims=None,ms=50):
        '''delta ccd_mag versus delta skymag_inap'''
        FS=25
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0.)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
            keep= self.legacy.stars.filter == band
            if np.where(keep)[0].size > 0:
                x= self.legacy.stars.skymag_inap[keep] - self.idl.stars.skymag_inap[keep]
                y= self.legacy.stars.ccd_mag[keep] - self.idl.stars.ccd_mag[keep]
                myscatter(ax[row],x,y,
                          color=color,m='o',s=ms,alpha=0.75)
                # lines through 0
                ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                ax[row].axvline(0,color='k',linestyle='dashed',linewidth=1)
                # fit line
                #try:
                #    popt= np.polyfit(x,y, 1)
                #except:
                #    raise ValueError
                #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                # slope 1
                #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                # Label
                ylab=ax[row].set_ylabel('%s ccd_mag (legacy - idl)' % (band,),fontsize=FS)
                if xlims:
                    ax[row].set_xlim(xlims)
                if ylims:
                    ax[row].set_ylim(ylims)
        xlab=ax[2].set_xlabel("skymag_inap (legacy - idl)",fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_ccdmag_skymaginap_%s.png' % (self.camera,prefix)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def star_plots_dmag_div_err(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        for leg_key,idl_key in zip(['apflux'],
                                   ['apflux']):
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.stars.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.ccd_mag[keep]
                    y= self.legacy.stars.get(leg_key)[keep] - self.idl.stars.get(idl_key)[keep] 
                    stddev= np.sqrt(self.legacy.stars.apflux[keep] +\
                                    self.idl.stars.apflux[keep] +\
                                    self.legacy.stars.apskyflux[keep] +\
                                    self.idl.stars.apskyflux[keep])
                    myscatter(ax[row],x,y / stddev,
                              color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s %s (legacy - idl) / sqrt(var)' % (band,idl_key),fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
            xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_dmag_diverr_%s_%s.png' % (self.camera,leg_key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def star_plots_dmag_div_err_sky(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        for leg_key,idl_key in zip(['apskyflux'],
                                   ['apskyflux']):
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.stars.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.ccd_mag[keep]
                    y= self.legacy.stars.get(leg_key)[keep] - self.idl.stars.get(idl_key)[keep] 
                    stddev= np.sqrt(self.legacy.stars.apskyflux[keep] +\
                                    self.idl.stars.apskyflux[keep])
                    myscatter(ax[row],x,y / stddev,
                              color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s %s (legacy - idl) / sqrt(var)' % (band,idl_key),fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
            xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_dmag_diverr_sky_%s_%s.png' % (self.camera,leg_key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def plots_apskyflux_vs_apskyflux(self,prefix=''):
        FS=25
        eFS=FS+5
        tickFS=FS
        xlims,ylims= None,None
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.5,wspace=0)
        for row,band,color in zip( range(3), self.fid.bands, ['g','r','m']):
            for leg_key,idl_key in zip(['apskyflux'],
                                             ['apskyflux']):
                keep= self.legacy.data.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.apskyflux[keep] #/ np.sqrt(self.idl.stars.apskyflux[keep])
                    y= self.legacy.stars.apskyflux[keep] #/ np.sqrt(self.legacy.stars.apskyflux[keep]) 
                    myscatter(ax[row],x,y,
                              color=color,m='o',s=20.,alpha=0.75, 
                              label='%s' % leg_key)
                    # fit line
                    popt= np.polyfit(x,y, 1)
                    x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    ax[row].plot(x,popt[0]*x + popt[1],c=color,ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
            ylab=ax[row].set_ylabel('%s %s (legacy)' % (band,leg_key),fontsize=FS)
        xlab=ax[2].set_xlabel("%s (idl)" % idl_key,fontsize=FS) #0.45'' galaxy
        for row in range(3)[::-1]:
            ax[row].tick_params(axis='both', labelsize=tickFS)
            leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_apskyflux2_%s.png' % (self.camera,prefix)
        plt.savefig(savefn,bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 



    def star_plots_dmag_err_bars(self,prefix='',xlims=None,ylims=None,ms=50):
        FS=25
        eFS=FS+5
        tickFS=FS
        for leg_key,idl_key in zip(['apflux'],
                                   ['apflux']):
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            for row,band,color in zip( range(3), self.fid.bands, ['g','r','m'] ):
                keep= self.legacy.stars.filter == band
                if np.where(keep)[0].size > 0:
                    x= self.idl.stars.ccd_mag[keep]
                    y= self.legacy.stars.get(leg_key)[keep] - self.idl.stars.get(idl_key)[keep] 
                    stddev= np.sqrt(self.legacy.stars.apflux[keep] +\
                                    self.idl.stars.apflux[keep] +\
                                    self.legacy.stars.apskyflux[keep] +\
                                    self.idl.stars.apskyflux[keep])
                    myerrorbar(ax[row],x,y,yerr=stddev,
                               color=color,m='o',s=ms,alpha=0.75)
                    # lines through 0
                    ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
                    # fit line
                    #try:
                    #    popt= np.polyfit(x,y, 1)
                    #except:
                    #    raise ValueError
                    #x= np.linspace(ax[row].get_xlim()[0],ax[row].get_xlim()[1],num=2)
                    #ax[row].plot(x,popt[0]*x + popt[1],c="gray",ls='--',label='m=%.2f, b=%.2f' % (popt[0],popt[1]))
                    # slope 1
                    #ax[row].plot([x.min(),x.max()],[x.min(),x.max()],'k--',lw=2)
                    # Label
                    ylab=ax[row].set_ylabel('%s %s (legacy - idl) / sqrt(var)' % (band,idl_key),fontsize=FS)
                    if xlims:
                        ax[row].set_xlim(xlims)
                    if ylims:
                        ax[row].set_ylim(ylims)
            xlab=ax[2].set_xlabel("ccd_mag (idl)",fontsize=FS) #0.45'' galaxy
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            savefn='legacy_vs_idl_%s_dmag_errbar_%s_%s.png' % (self.camera,leg_key,prefix)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 



    def plot_star_sn(self):
        FS=25
        eFS=FS+5
        tickFS=FS
        xlims,ylims= None,None
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        sn={}
        sn['idl']= self.idl.stars.apflux/ np.sqrt(self.idl.stars.apskyflux)
        sn['idl_unm']= self.idl.stars_unm.apflux/ np.sqrt(self.idl.stars_unm.apskyflux)
        sn['leg']= self.legacy.stars.apflux/ np.sqrt(self.legacy.stars.apskyflux)
        sn['leg_unm']= self.legacy.stars_unm.apflux/ np.sqrt(self.legacy.stars_unm.apskyflux)
        for row,band in zip( range(3), self.fid.bands):
            # Legacy
            color='b'
            keep= self.legacy.stars.filter == band
            if np.where(keep)[0].size > 0:
                myhist_step(ax[row],sn['leg'][keep],bins=50,color=color,ls='solid',
                            normed=False,lw=2,label='leg matched (%s)' % len(sn['leg'][keep]))
            # unm
            keep= self.legacy.stars_unm.filter == band
            if np.where(keep)[0].size > 0:
                myhist_step(ax[row],sn['leg_unm'][keep],bins=50,color=color,ls='dashed',
                            normed=False,lw=2,label='leg unm (%s)' % len(sn['leg_unm'][keep]) )
            # IDL
            color='g'
            keep= self.idl.stars.filter == band
            if np.where(keep)[0].size > 0:
                myhist_step(ax[row],sn['idl'][keep],bins=50,color=color,ls='solid',
                            normed=False,lw=2,label='idl matched (%s)' % len(sn['idl'][keep]))
            # unm
            keep= self.idl.stars_unm.filter == band
            if np.where(keep)[0].size > 0:
                myhist_step(ax[row],sn['idl_unm'][keep],bins=50,color=color,ls='dashed',
                            normed=False,lw=2,label='idl unm (%s)' % len(sn['idl_unm'][keep]))
            # Label
            ylab=ax[row].set_ylabel('N',fontsize=FS)
            ax[row].set_xscale('log')
        xlab=ax[2].set_xlabel("S/N",fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,fontsize=FS-5)
        savefn='legacy_vs_idl_%s_sn_logx.png' % (self.camera,)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        print "wrote %s" % savefn 
        for row in range(3):
            ax[row].set_yscale('log')
        savefn='legacy_vs_idl_%s_sn_logxlogy.png' % (self.camera,)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 


    def plot_star_gaiaoffset(self):
        FS=25
        eFS=FS+5
        tickFS=FS
        xlims,ylims= None,None
        fig,ax= plt.subplots(3,1,figsize=(10,15))
        plt.subplots_adjust(hspace=0.5,wspace=0.)
        sn={}
        for row,band in zip( range(3), self.fid.bands):
            # IDL
            color='g'
            keep= self.idl.stars.filter == band
            if np.where(keep)[0].size > 0:
                x= self.idl.stars.raoff
                y= self.idl.stars.decoff
                myscatter(ax[row],x[keep],y[keep], color=color,s=20.,alpha=0.75,
                          label='idl matched (%s)' % len(x[keep]))
            # unm
            keep= self.idl.stars_unm.filter == band
            if np.where(keep)[0].size > 0:
                x= self.idl.stars_unm.raoff
                y= self.idl.stars_unm.decoff
                myscatter_open(ax[row],x[keep],y[keep], color=color,s=20.,alpha=0.75,
                               label='idl unm (%s)' % len(x[keep]))
            # Legacy
            color='b'
            keep= self.legacy.stars.filter == band
            if np.where(keep)[0].size > 0:
                x= self.legacy.stars.radiff
                y= self.legacy.stars.decdiff
                myscatter(ax[row],x[keep],y[keep], color=color,s=20.,alpha=0.75,
                          label='leg matched (%s)' % len(x[keep]))
            # unm
            keep= self.legacy.stars_unm.filter == band
            if np.where(keep)[0].size > 0:
                x= self.legacy.stars_unm.radiff
                y= self.legacy.stars_unm.decdiff
                myscatter_open(ax[row],x[keep],y[keep], color=color,s=20.,alpha=0.75,
                               label='leg unm (%s)' % len(x[keep]))
            # lines through 0
            ax[row].axhline(0,color='k',linestyle='dashed',linewidth=1)
            ax[row].axvline(0,color='k',linestyle='dashed',linewidth=1)
            # Label
            ylab=ax[row].set_ylabel('Dec Offset (arcsec)',fontsize=FS)
            #leg=ax[row].legend(loc=(0.,1.02),ncol=3,scatterpoints=1,markerscale=3,fontsize=FS-5)
            ax[row].set_xlim(-0.1,0.1)
            ax[row].set_ylim(-0.1,0.1)
            ax[row].set_aspect('equal')
        xlab=ax[2].set_xlabel("Ra Offset (arcsec)",fontsize=FS) #0.45'' galaxy
        for row in range(3):
            ax[row].tick_params(axis='both', labelsize=tickFS)
        savefn='legacy_vs_idl_%s_gaiaoffset.png' % (self.camera,)
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

       
#    def plot_on_ccd(ccdname='N19'):
#        rad = 7./self.fid.pixscale
#        # CCD filename
#        ccd_fn= self.legacy.stars.image_filename[(self.legacy.data.expid == expid)*\
#                                                 (self.legacy.data.ccdname == ccdname)]
#        hdu= self.legacy.stars.image_hdu[(self.legacy.data.expid == expid)*\
#                                         (self.legacy.data.ccdname == ccdname)]
#        ccd_fn= os.path.join('/project/projectdirs/cosmo/staging',ccd_fn)
#        img,h = fitsio.read(ccd_fn, ext=hdu, header=True)
#        sz = img.shape #img.size
#        #draw = ImageDraw.Draw(img)
#        plt.imshow(img, interpolation='nearest', origin='lower',
#                        cmap='gray', ticks=True,
#                   vmin=np.percentile(img,q=5), vmax=vmin=np.percentile(img,q=95))
#        # CCD only
#        cuts={}
#        cuts['idl']= (self.idl.stars.ccdname == ccdname)
#        cuts['leg']= (self.legacy.stars.ccdname == ccdname)
#        # Matched * on ccd
#        tab= self.idl.stars.copy()[self.idl.stars.ccdname == ccdname]
#        [draw.ellipse((cat.x-rad, sz[1]-cat.x-rad, 
#                       cat.x+rad, sz[1]-cat.y+rad), outline='yellow') for cat in tab]
#        # Legacy only * on ccd
#        tab= self.legacy.stars_unm.copy()[self.legacy_unm.stars.ccdname == ccdname]
#        [draw.ellipse((cat.x-rad, sz[1]-cat.x-rad, 
#                       cat.x+rad, sz[1]-cat.y+rad), outline='yellow') for cat in tab]
#        # IDL only * on ccd
#        tab= self.idl.stars_unm.copy()[self.idl_unm.stars.ccdname == ccdname]
#        [draw.ellipse((cat.x-rad, sz[1]-cat.x-rad, 
#                       cat.x+rad, sz[1]-cat.y+rad), outline='yellow') for cat in tab]
#        # Save
#        im.save(qafile)

#######
# Residual plots of legacy zeropoints - arjuns
# legacy zeropoints repackaged to arjuns colnames and units
# -zpt & -star files
#######

class ZeropointResiduals(object):
    '''
    use create_zeropoint_table() to convert my -zpt.fits table into 
        arjuns zeropoint-*.fits table
    then run this on my converted table versus arjuns
    '''
    def __init__(self,legacy_zpt_fn,arjun_zpt_fn):
        self.legacy= fits_table(legacy_zpt_fn)
        self.idl= fits_table(arjun_zpt_fn)
        self.match()
        self.plot_residuals(doplot='diff')

    def match(self):
        m1, m2, d12 = match_radec(self.legacy.ccdra,self.legacy.ccddec,
                                  self.idl.ccdra,self.idl.ccddec,1./3600.0,nearest=True)
        self.legacy= self.legacy[m1]
        self.idl= self.idl[m2]
    
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

    def get_defaultdict_ylim(self,doplot,ylim=None):
        ylim_dict=defaultdict(lambda: ylim)
        if doplot == 'diff':
            ylim_dict['ccdra']= (-1e-8,1e-8)
            ylim_dict['ccddec']= ylim_dict['ra']
            ylim_dict['ccdzpt']= (-0.005,0.005)
        elif doplot == 'div':
            ylim_dict['ccdzpt']= (0.995,1.005)
            ylim_dict['ccdskycounts']= (0.99,1.01)
        else:
            pass
        return ylim_dict
 
    def get_defaultdict_xlim(self,doplot,xlim=None):
        xlim_dict=defaultdict(lambda: xlim)
        if doplot == 'diff':
            pass
        elif doplot == 'div':
            pass
        else:
            pass
        return xlim_dict      

    def plot_residuals(self,doplot=None,
                       ms=100,use_keys=None,ylim=None,xlim=None):
        '''two plots of everything numberic between legacy zeropoints and Arjun's
        1) x vs. y-x 
        2) x vs. y/x
        use_keys= list of legacy_keys to use ONLY
        '''
        #raise ValueError
        assert(doplot in ['diff','div'])
        # All keys and any ylims to use
        cols= self.get_numeric_keys()
        ylim= self.get_defaultdict_ylim(doplot=doplot,ylim=ylim)
        xlim= self.get_defaultdict_xlim(doplot=doplot,xlim=xlim)
        # Plot
        FS=25
        eFS=FS+5
        tickFS=FS
        bands= set(self.legacy.filter)
        ccdnums= set(self.legacy.ccdnum)
        for cnt,col in enumerate(cols):
            if use_keys:
                if not col in use_keys:
                    print('skipping col=%s' % col)
                    continue
            # Plot
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            # Loop over bands, hdus
            for row,band in zip( range(3), bands ):
                for ccdnum,color in zip(ccdnums,['g','r','m','b','k','y']*12):
                    keep= (self.legacy.filter == band)*\
                          (self.legacy.ccdnum == ccdnum)
                    if np.where(keep)[0].size > 0:
                        x= self.idl.get( col )[keep]
                        xlabel= 'IDL'
                        if doplot == 'diff':
                            y= self.legacy.get(col)[keep] - self.idl.get(col)[keep]
                            y_horiz= 0
                            ylabel= 'Legacy - IDL'
                        elif doplot == 'div':
                            y= self.legacy.get(col)[keep] / self.idl.get(col)[keep]
                            y_horiz= 1
                            ylabel= 'Legacy / IDL'
                        myscatter(ax[row],x,y,color=color,m='o',s=ms,alpha=0.75) 
                        ax[row].axhline(y_horiz,color='k',linestyle='dashed',linewidth=1)
                        #ax[row].text(0.025,0.88,idl_key,\
                        #             va='center',ha='left',transform=ax[cnt].transAxes,fontsize=20)
                        ylab= ax[row].set_ylabel(ylabel,fontsize=FS)
            xlab = ax[row].set_xlabel(xlabel,fontsize=FS)
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                if ylim[col]:
                    ax[row].set_ylim(ylim[col])
                if xlim[col]:
                    ax[row].set_xlim(xlim[col])
            savefn="zeropointresiduals_%s_%s.png" % (doplot,col)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

class MatchesResiduals(object):
    '''
    use create_matches_table() to convert my -star.fits table into 
        arjuns matches-*.fits table
    then run this on my converted table versus arjuns
    '''
    def __init__(self,legacy_fn,arjun_fn):
        self.legacy= fits_table(legacy_fn)
        self.idl= fits_table(arjun_fn)
        self.match()
        self.plot_residuals(doplot='diff')

    def match(self):
        m1, m2, d12 = match_radec(self.legacy.ccd_ra,self.legacy.ccd_dec,
                                  self.idl.ccd_ra,self.idl.ccd_dec,1./3600.0,nearest=True)
        self.legacy= self.legacy[m1]
        self.idl= self.idl[m2]
    
    def get_numeric_keys(self):
        idl_keys= \
            ['ccd_x','ccd_y','ccd_ra','ccd_dec',
             'ccd_mag','ccd_sky',
             'raoff','decoff',
             'magoff',
             'nmatch',
             'gmag','ps1_g','ps1_r','ps1_i','ps1_z']
        return idl_keys

    def get_defaultdict_ylim(self,doplot,ylim=None):
        ylim_dict=defaultdict(lambda: ylim)
        if doplot == 'diff':
            ylim_dict['ccd_ra']= (-1e-8,1e-8)
            ylim_dict['ccd_dec']= ylim_dict['ra']
        elif doplot == 'div':
            pass
        else:
            pass
        return ylim_dict
 
    def get_defaultdict_xlim(self,doplot,xlim=None):
        xlim_dict=defaultdict(lambda: xlim)
        if doplot == 'diff':
            pass
        elif doplot == 'div':
            pass
        else:
            pass
        return xlim_dict      

    def plot_residuals(self,doplot=None,
                       ms=100,use_keys=None,ylim=None,xlim=None):
        '''two plots of everything numberic between legacy zeropoints and Arjun's
        1) x vs. y-x 
        2) x vs. y/x
        use_keys= list of legacy_keys to use ONLY
        '''
        #raise ValueError
        assert(doplot in ['diff','div'])
        # All keys and any ylims to use
        cols= self.get_numeric_keys()
        ylim= self.get_defaultdict_ylim(doplot=doplot,ylim=ylim)
        xlim= self.get_defaultdict_xlim(doplot=doplot,xlim=xlim)
        # Plot
        FS=25
        eFS=FS+5
        tickFS=FS
        bands= set(self.legacy.filter)
        ccdnums= set(self.legacy.image_hdu)
        for cnt,col in enumerate(cols):
            if use_keys:
                if not col in use_keys:
                    print('skipping col=%s' % col)
                    continue
            # Plot
            fig,ax= plt.subplots(3,1,figsize=(10,15))
            plt.subplots_adjust(hspace=0.2,wspace=0.)
            # Loop over bands, hdus
            for row,band in zip( range(3), bands ):
                for ccdnum,color in zip(ccdnums,['g','r','m','b','k','y']*12):
                    keep= (self.legacy.filter == band)*\
                          (self.legacy.ccdnum == ccdnum)
                    if np.where(keep)[0].size > 0:
                        x= self.idl.get( col )[keep]
                        xlabel= 'IDL'
                        if doplot == 'diff':
                            y= self.legacy.get(col)[keep] - self.idl.get(col)[keep]
                            y_horiz= 0
                            ylabel= 'Legacy - IDL'
                        elif doplot == 'div':
                            y= self.legacy.get(col)[keep] / self.idl.get(col)[keep]
                            y_horiz= 1
                            ylabel= 'Legacy / IDL'
                        myscatter(ax[row],x,y,color=color,m='o',s=ms,alpha=0.75) 
                        ax[row].axhline(y_horiz,color='k',linestyle='dashed',linewidth=1)
                        #ax[row].text(0.025,0.88,idl_key,\
                        #             va='center',ha='left',transform=ax[cnt].transAxes,fontsize=20)
                        ylab= ax[row].set_ylabel(ylabel,fontsize=FS)
            xlab = ax[row].set_xlabel(xlabel,fontsize=FS)
            for row in range(3):
                ax[row].tick_params(axis='both', labelsize=tickFS)
                if ylim[col]:
                    ax[row].set_ylim(ylim[col])
                if xlim[col]:
                    ax[row].set_xlim(xlim[col])
            savefn="matchesresiduals_%s_%s.png" % (doplot,col)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

def band2color(band):
    d=dict(g='g',r='r',z='m')
    return d[band]

def col2plotname(key):
    d= dict(airmass= 'Airmass',
            fwhm= 'FWHM (pixels)',
            seeing= 'FWHM (arcsec)',
            gain= 'Gain (e/ADU)',
            skymag= 'Sky Brightness (AB mag/arcsec^2)',
            skyrms= 'Sky RMS (e/sec/pixel)',
            transp= 'Atmospheric Transparency',
            zpt= 'Zeropoint (e/s)',
            err_on_radecoff= 'Astrometric Offset Error (Std Err of Median, arcsec)',
            err_on_phoff= r'Zeropoint Error (Std Err of Median), mag)',
            psfdepth= r'Point-source Single Pass Depth (5$\sigma$)',
            galdepth= r'Galaxy Single Pass Depth (5$\sigma$)')
    d['psfdepth_extcorr']= r'Point-source Depth (5$\sigma$, ext. corrected)'
    d['galdepth_extcorr']= r'Galaxy Depth (5$\sigma$, ext. corrected)'
    return d.get(key,key)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                            description='Generate a legacypipe-compatible CCDs file \
                                        from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--leg_dir',action='store',default='/global/cscratch1/sd/kaylanb/kaylan_Test',required=False)
    parser.add_argument('--idl_dir',action='store',default='/global/cscratch1/sd/kaylanb/arjundey_Test/AD_exact_skymed',required=False)
    args = parser.parse_args()

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
