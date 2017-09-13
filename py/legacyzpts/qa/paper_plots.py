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

mygray='0.6'

def myhist(ax,data,bins=20,color='b',normed=False,ls='-',label=None):
    if label:
        _=ax.hist(data,bins=bins,facecolor=color,normed=normed,
                  histtype='stepfilled',edgecolor='none', alpha=0.75,label=label)
    else:
        _=ax.hist(data,bins=bins,facecolor=color,normed=normed,
                  histtype='stepfilled',edgecolor='none', alpha=0.75)

def myhist_step(ax,data,bins=20,color='b',normed=False,lw=2,ls='solid',label=None,
                return_vals=False):
    if label:
        h,bins,_=ax.hist(data,bins=bins,color=color,normed=normed,
                  histtype='step',lw=lw,ls=ls,label=label)
    else:
        h,bins,_=ax.hist(data,bins=bins,color=color,normed=normed,
                  histtype='step',lw=lw,ls=ls)
    if return_vals:
        return h,bins


def myscatter(ax,x,y, color='b',m='o',s=10.,alpha=0.75,label=None):
    if label is None or label == 'None':
        ax.scatter(x,y, c=color,edgecolors='none',marker=m,s=s,rasterized=True,alpha=alpha)
    else:
        ax.scatter(x,y, c=color,edgecolors='none',marker=m,s=s,rasterized=True,alpha=alpha,label=label)

def myscatter_open(ax,x,y, color='b',m='o',s=10.,alpha=0.75,label=None):
    if label is None or label == 'None':
        ax.scatter(x,y, facecolors='none',edgecolors=color,marker=m,s=s,rasterized=True,alpha=alpha)
    else:
        ax.scatter(x,y, facecolors='none',edgecolors=color,marker=m,s=s,rasterized=True,alpha=alpha,label=label)

def myannot(ax,xarr,yarr,sarr, ha='left',va='bottom',fontsize=20):
    '''x,y,s are iterable'''
    for x,y,s in zip(xarr,yarr,sarr):
        ax.annotate(s,xy=(x,y),xytext=(x,y),
                    horizontalalignment=ha,verticalalignment=va,fontsize=fontsize)

def mytext(ax,x,y,text, ha='left',va='center',fontsize=20):
    '''adds text in x,y units of fraction axis'''
    ax.text(x,y,text, horizontalalignment=ha,verticalalignment=va,
            transform=ax.transAxes,fontsize=fontsize)


def myerrorbar(ax,x,y, yerr=None,xerr=None,color='b',ls='none',m='o',s=10.,mew=1,alpha=0.75,label=None):
    if label is None or label == 'None':
        ax.errorbar(x,y, xerr=xerr,yerr=yerr,ls=ls,alpha=alpha,
                    marker=m,ms=s,mfc=color,mec=color,ecolor=color,mew=mew)
    else:
        ax.errorbar(x,y, xerr=xerr,yerr=yerr,ls=ls,alpha=alpha,label=label,
                    marker=m,ms=s,mfc=color,mec=color,ecolor=color,mew=mew)
        #ax.errorbar(x,y, xerr=xerr,yerr=yerr, c=color,ecolor=color,marker=m,s=s,rasterized=True,alpha=alpha,label=label)

def getrms(x):
    return np.sqrt( np.mean( np.power(x,2) ) ) 


class LegacyZpts(object):
    '''
    Gets all the data we need: legacy-zeropoints for DECam, additional data like exp needed
    Applies appropriate cuts etc

    USE:
    ----------
    L= LegacyZpts(zpt_list='/global/cscratch1/sd/kaylanb/cosmo/staging/decam//DECam_CP/allzpts_list_2017-01-11.txt')
    if os.path.exists(L.pkl_fn):
        with open(L.pkl_fn,'r') as foo:
            L,= pickle.load(foo)
    else:
        L.get_data() 
    -----------

    self.data -- data
    self.obs_method -- observing method, dict of "dynamic,fixed"
    '''
    def __init__(self,camera=None,outdir=None,what='zpt_data'):
        '''
        '''
        # Outdir
        if outdir:
            self.outdir= outdir
        else:
            self.outdir='/global/cscratch1/sd/kaylanb/observing_paper_zptfixes'
        #self.outdir='/global/cscratch1/sd/kaylanb/observing_paper'
        assert(camera in ['decam','mosaic'])
        self.camera= camera
        self.fid= params.get_fiducial(camera=self.camera)
        self.zpt_list= self.get_zptlist(what)
        # Fits table having all zpts
        self.merged_zpts= self.zpt_list.replace('.txt','_merged.fits')
        self.final_zpts= self.zpt_list.replace('.txt','_final.fits')
        # Get data
        self.get_raw_zpts()

    def get_zptlist(self,what='zpt_data'):
        if what == 'zpt_data':
            return os.path.join(self.outdir,'%s_list_zpts.txt' % self.camera)
        elif what == 'star_data':
            return os.path.join(self.outdir,'%s_list_stars.txt' % self.camera)
        else: raise ValueError('%s not supported' % what)


    def get_raw_zpts(self):
        '''Read in merged_zpts, apply all cuts, any fixes, and add tneed,depth quantities 
        '''
        if os.path.exists( self.merged_zpts ):
            # Read merged fits
            self.load('merged')
        else: 
            # Merge 
            data=[]
            data_fns= np.loadtxt(self.zpt_list,dtype=str) 
            if data_fns.size == 1:
                data_fns= [data_fns.tostring()]
            for cnt,fn in enumerate(data_fns):
                print('%d/%d: ' % (cnt+1,len(data_fns)))
                try:
                    tb= fits_table(fn) 
                    # If here, was able to read all 4 tables, store in Big Table
                    data.append( tb ) 
                except IOError:
                    print('WARNING: cannot read: %s' % fn)
            self.data= merge_tables(data, columns='fillzero')
            self.save('merged')
        print('Merged zpt data: zpts=%d' % self.num_zpts())

    def load(self,name):
        assert(name in ['merged','final'])
        if name == 'merged':
            fn= self.merged_zpts
        elif name == 'final':
            fn= self.final_zpts
        self.data= fits_table(fn)
        print('Loaded %s' % fn)

    def save(self,name):
        assert(name in ['merged','final'])
        if name == 'merged':
            fn= self.merged_zpts
        elif name == 'final':
            fn= self.final_zpts
        self.data.writeto(fn)
        print('Wrote %s' % fn)
 
    def num_zpts(self):
        return len(self.data)

#######
# Master depth calculator, uses non galdepth columns to get exactly galdepth in annotated-ccds
#######

class Depth(object):
    def __init__(self,camera,skyrms,gain,fwhm,zpt):
        '''
        See: observing_paper/depth.py
        Return depth that agrees with galdepth doing equiv with annotated ccd cols
        skyrms -- e/s
        gain -- e/ADU
        fwhm -- pixels so that gaussian std in pixels = fwhm/2.35
        zpt -- e/s
        '''
        assert(camera in ['decam','mosaic'])
        self.camera = camera
        self.skyrms= skyrms.copy() #Will modify to get in right units
        self.gain= gain.copy()
        self.fwhm= fwhm.copy()
        self.zpt= zpt.copy()
        if self.camera == 'decam':
            # natural camera units ADU/sec
            self.skyrms /= self.gain # e/sec --> ADU/sec
            self.zpt -= 2.5*np.log10(self.gain) # e/sec --> ADU/sec
        elif self.camera == 'mosaic':
            # e/sec are the natual camera units
            pass
        # self.get_depth_legacy_zpts(which='gal')

    def get_depth_legacy_zpts(self,which=None):
        assert(which in ['gal','psf'])
        self.which= which
        sigma= self.sigma_legacy_zpts() 
        return -2.5*np.log10(5 * sigma) + self.zpt #natural camera units

    def sigma_legacy_zpts(self):
        return self.skyrms *np.sqrt(self.neff_empirical())

    def neff_empirical(self):
        #see observing_paper/depth.py plots
        if self.which == 'psf':
            rhalf=0
            if self.camera == 'decam':
                slope= 1.23 
                yint= 9.43
            elif self.camera == 'mosaic':
                slope= 1.41 
                yint= 1.58
        elif self.which == 'gal':
            rhalf=0.45
            if self.camera == 'decam':
                slope= 1.24 
                yint= 45.86
            elif self.camera == 'mosaic':
                slope= 1.48 
                yint= 35.78
        return slope * self.neff_15(rhalf=rhalf) + yint

    def neff_15(self,rhalf=0.45,pix=0.262):
        '''seeing = FWHM/2.35 where FWHM is in units of Pixels'''
        seeing= self.fwhm / 2.35
        return 4*np.pi*seeing**2 + 8.91*rhalf**2 + pix**2/12  



class DepthRequirements(object):
    def __init__(self):
        self.desi= self.depth_requirement_dict()
    
    def get_single_pass_depth(self,band,which,camera):
        assert(which in ['gal','psf'])
        assert(camera in ['decam','mosaic'])
        # After 1 pass
        if camera == 'decam':
            return self.desi[which][band] - 2.5*np.log10(2)
        elif camera == 'mosaic':
            return self.desi[which][band] - 2.5*np.log10(3)

    def depth_requirement_dict(self):
        mags= defaultdict(lambda: defaultdict(dict))
        mags['gal']['g']= 24.0
        mags['gal']['r']= 23.4
        mags['gal']['z']= 22.5
        mags['psf']['g']= 24.7
        mags['psf']['r']= 23.9
        mags['psf']['z']= 23.0
        return mags


#######
# Master depth calculator, uses non galdepth columns to get exactly galdepth in annotated-ccds
#######

class Depth(object):
    def __init__(self,camera,skyrms,gain,fwhm,zpt):
        '''
        See: observing_paper/depth.py
        Return depth that agrees with galdepth doing equiv with annotated ccd cols
        skyrms -- e/s
        gain -- e/ADU
        fwhm -- pixels so that gaussian std in pixels = fwhm/2.35
        zpt -- e/s
        '''
        assert(camera in ['decam','mosaic'])
        self.camera = camera
        self.skyrms= skyrms.copy() #Will modify to get in right units
        self.gain= gain.copy()
        self.fwhm= fwhm.copy()
        self.zpt= zpt.copy()
        if self.camera == 'decam':
            # natural camera units ADU/sec
            self.skyrms /= self.gain # e/sec --> ADU/sec
            self.zpt -= 2.5*np.log10(self.gain) # e/sec --> ADU/sec
        elif self.camera == 'mosaic':
            # e/sec are the natual camera units
            pass
        # self.get_depth_legacy_zpts(which='gal')

    def get_depth_legacy_zpts(self,which=None):
        assert(which in ['gal','psf'])
        self.which= which
        sigma= self.sigma_legacy_zpts() 
        return -2.5*np.log10(5 * sigma) + self.zpt #natural camera units

    def sigma_legacy_zpts(self):
        return self.skyrms *np.sqrt(self.neff_empirical())

    def neff_empirical(self):
        #see observing_paper/depth.py plots
        if self.which == 'psf':
            rhalf=0
            if self.camera == 'decam':
                slope= 1.23 
                yint= 9.43
            elif self.camera == 'mosaic':
                slope= 1.41 
                yint= 1.58
        elif self.which == 'gal':
            rhalf=0.45
            if self.camera == 'decam':
                slope= 1.24 
                yint= 45.86
            elif self.camera == 'mosaic':
                slope= 1.48 
                yint= 35.78
        return slope * self.neff_15(rhalf=rhalf) + yint

    def neff_15(self,rhalf=0.45,pix=0.262):
        '''seeing = FWHM/2.35 where FWHM is in units of Pixels'''
        seeing= self.fwhm / 2.35
        return 4*np.pi*seeing**2 + 8.91*rhalf**2 + pix**2/12  

class DepthRequirements(object):
    def __init__(self):
        self.desi= self.depth_requirement_dict()
    
    def get_single_pass_depth(self,band,which,camera):
        assert(which in ['gal','psf'])
        assert(camera in ['decam','mosaic'])
        # After 1 pass
        if camera == 'decam':
            return self.desi[which][band] - 2.5*np.log10(2)
        elif camera == 'mosaic':
            return self.desi[which][band] - 2.5*np.log10(3)

    def depth_requirement_dict(self):
        mags= defaultdict(lambda: defaultdict(dict))
        mags['gal']['g']= 24.0
        mags['gal']['r']= 23.4
        mags['gal']['z']= 22.5
        mags['psf']['g']= 24.7
        mags['psf']['r']= 23.9
        mags['psf']['z']= 23.0
        return mags

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



#######
# Histograms of CCD statistics from legacy zeropoints
# for paper
#######
# a= ZeropointHistograms('/path/to/combined/decam-zpt.fits','path/to/mosaics-zpt.fits')

def err_on_radecoff(rarms,decrms,nmatch):
    err_raoff= 1.253 * rarms / np.sqrt(nmatch)
    err_decoff= 1.253 * decrms / np.sqrt(nmatch)
    return np.sqrt(err_raoff**2 + err_decoff**2)

class ZeropointHistograms(object):
    '''Histograms for papers'''
    def __init__(self,decam_zpts=None,mosaic_zpts=None):
        self.decam= decam_zpts
        self.mosaic= mosaic_zpts
        if self.decam:
            self.decam= fits_table(self.decam)
        if self.mosaic:
            self.mosaic= fits_table(self.mosaic)
        self.add_keys()
        self.num_exp= self.store_num_exp()
        #self.plot_hist_1d()
        #self.plot_2d_scatter()
        #self.plot_astro_photo_scatter()
        self.plot_hist_depth(legend=False)
        self.print_ccds_table()
        self.print_requirements_table()
   
    def add_keys(self):
        if self.decam:
            self.decam.set('seeing',self.decam.fwhm * 0.262)
            self.set_Aco_EBV(self.decam,camera='decam')
            self.decam.set('err_on_phoff',1.253 * self.decam.phrms/np.sqrt(self.decam.nmatch)) # error on median
            self.decam.set('err_on_radecoff',err_on_radecoff(self.decam.rarms,self.decam.decrms,
                                                             self.decam.nmatch))
            # Depths
            depth_obj= Depth('decam',
                             self.decam.skyrms,self.decam.gain,
                             self.decam.fwhm,self.decam.zpt)
            self.decam.set('psfdepth', depth_obj.get_depth_legacy_zpts('psf'))
            self.decam.set('galdepth', depth_obj.get_depth_legacy_zpts('gal'))
            for col in ['psfdepth','galdepth']:
                self.decam.set(col+'_extcorr', self.decam.get(col) - self.decam.AcoEBV)
        if self.mosaic:
            self.mosaic.set('seeing',self.mosaic.fwhm * 0.26)
            self.set_Aco_EBV(self.mosaic,camera='mosaic')
            self.mosaic.set('err_on_phoff',1.253 * self.mosaic.phrms/np.sqrt(self.mosaic.nmatch)) # error on median
            self.mosaic.set('err_on_radecoff',err_on_radecoff(self.mosaic.rarms,self.mosaic.decrms,
                                                              self.mosaic.nmatch))
            # Depths
            depth_obj= Depth('mosaic',
                             self.mosaic.skyrms,self.mosaic.gain,
                             self.mosaic.fwhm,self.mosaic.zpt)
            self.mosaic.set('psfdepth', depth_obj.get_depth_legacy_zpts('psf'))
            self.mosaic.set('galdepth', depth_obj.get_depth_legacy_zpts('gal'))
            for col in ['psfdepth','galdepth']:
                self.mosaic.set(col+'_extcorr', self.mosaic.get(col) - self.mosaic.AcoEBV)
 
    def store_num_exp(self):
        num={}
        num['mosaic_z']= len(set(self.mosaic.expnum))
        for band in ['g','r','z']:
            keep= self.decam.filter == band
            num['decam_%s' % band]= len(set(self.decam.expnum[keep]))
        return num

    def set_Aco_EBV(self,tab,camera=None):
        '''tab -- legacy zeropoints -zpt.fits table'''
        assert(camera in ['decam','mosaic'])
        # Look up E(B-V) in SFD map
        print('Loading SFD maps...')
        sfd = SFDMap()
        ebv= sfd.ebv(tab.ra, tab.dec) 
        data= np.zeros(len(tab))
        # Aco coeff
        if camera == 'decam':
            Aco= dict(g=3.214,r=2.165,z=1.562)
        elif camera == 'mosaic':
            Aco= dict(z=1.562)
        # Ext corr
        for band in set(tab.filter):
            keep= tab.filter == band
            data[keep]= Aco[band] * ebv[keep]
        assert(np.all(data > 0.))
        tab.set('AcoEBV',data)

    def get_numeric_keys(self):
        keys= \
            ['skymag','skyrms','zpt','airmass','fwhm','gain','transp',
             'seeing','psfdepth','galdepth']
        return keys

    def get_defaultdict_ylim(self,ylim=None):
        ylim_dict=defaultdict(lambda: ylim)
        ylim_dict['skymag']= (0,2.)
        ylim_dict['zpt']= (0,5.)
        ylim_dict['transp']= (0,9)
        return ylim_dict

    def get_defaultdict_xlim(self,xlim=None):
        xlim_dict=defaultdict(lambda: xlim)
        xlim_dict['skymag']= (17,22.5)
        xlim_dict['skyrms']= (0,1.5)
        xlim_dict['zpt']= (25.8,27.2)
        xlim_dict['airmass']= (0.9,2.5)
        xlim_dict['transp']= (0.5,1.4)
        xlim_dict['gain']= (0,5)
        xlim_dict['seeing']= (0.5,2.4)
        xlim_dict['fwhm']= (2,10)
        xlim_dict['psfdepth']= (21,25)
        xlim_dict['galdepth']= (21,25)
        return xlim_dict      

    def get_fiducial(self,key,camera,band):
        assert(camera in ['decam','mosaic'])
        d= {}
        X0, see0= 1.3, 1.3
        if camera == 'decam':
            d['zpt']= dict(g=26.610, r=26.818, z=26.484) # e/s
            d['skymag']= dict(g=22.04, r=20.91, z=18.46)
            d['airmass']= dict(g=X0,r=X0,z=X0) #arcsec
            d['seeing']= dict(g=see0,r=see0,z=see0) #arcsec
        elif camera == 'mosaic':
            d['zpt']= dict(z=26.552) # e/s
            d['skymag']= dict(z=18.46)
            d['airmass']= dict(z=X0) #arcsec
            d['seeing']= dict(z=see0) #arcsec
        return d[key][band]

    def plot_hist_1d(self):
        # All keys and any ylims to use
        cols= self.get_numeric_keys()
        ylim= self.get_defaultdict_ylim()
        xlim= self.get_defaultdict_xlim()
        fiducial_keys= ['seeing','zpt','skymag','airmass']
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        for key in cols:
            fig,ax= plt.subplots(figsize=(7,5))
            if xlim[key]:
                bins= np.linspace(xlim[key][0],xlim[key][1],num=40)
            else:
                bins=40
            # decam
            if self.decam:
                for band in set(self.decam.filter):
                    keep= self.decam.filter == band
                    myhist_step(ax,self.decam.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',
                                label='%s (DECam, %d)' % (band,self.num_exp['decam_'+band]))
                    if key in fiducial_keys:
                        ax.axvline(self.get_fiducial(key,'decam',band),
                                   c=band2color(band),ls='dotted',lw=1) 
                    
            # mosaic
            if self.mosaic:
                for band in set(self.mosaic.filter):
                    keep= self.mosaic.filter == band
                    myhist_step(ax,self.mosaic.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='dashed',
                                label='%s (Mosaic3, %d)' % (band,self.num_exp['mosaic_'+band]))
                    if key in fiducial_keys:
                        ls= 'dotted'
                        if key == 'zpt':
                            ls= 'dashed'
                        ax.axvline(self.get_fiducial(key,'mosaic',band),
                                   c=band2color(band),ls=ls,lw=1) 
            # Label
            ylab=ax.set_ylabel('PDF',fontsize=FS)
            ax.tick_params(axis='both', labelsize=tickFS)
            leg=ax.legend(loc=(0,1.02),ncol=2,fontsize=FS-2)
            if ylim[key]:
                ax.set_ylim(ylim[key])
            if xlim[key]:
                ax.set_xlim(xlim[key])
            xlab=ax.set_xlabel(col2plotname(key),fontsize=FS) #0.45'' galaxy
            savefn='hist_1d_%s.png' % key
            plt.savefig(savefn, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def plot_hist_depth(self,legend=True):
        # All keys and any ylims to use
        cols= ['psfdepth_extcorr','galdepth_extcorr']
        xlim= (21,24.5)
        ylim= None
        # Depths
        depth_obj= DepthRequirements()
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(2,2,figsize=(10,8))
        if legend:
            plt.subplots_adjust(hspace=0.2,wspace=0)
        else:
            plt.subplots_adjust(hspace=0,wspace=0)
        if xlim:
            bins= np.linspace(xlim[0],xlim[1],num=40)
        else:
            bins=40
        d_row=dict(decam=0,mosaic=1,bok=1)
        d_col=dict(psf=0,gal=1)
        for which in ['psf','gal']:
            key= which+'depth_extcorr'
            # decam
            if self.decam:
                row=d_row['decam']
                col=d_col[which]
                for band in set(self.decam.filter):
                    keep= self.decam.filter == band
                    myh, mybins= myhist_step(ax[row,col],self.decam.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',lw=2,
                                label='%s (DECam)' % band, return_vals=True)
                    # requirements
                    if which == 'gal':
                        p1_depth= depth_obj.get_single_pass_depth(band,which,'decam')
                        ax[row,col].axvline(p1_depth,
                                        c=band2color(band),ls='dashed',lw=1,
                                        label=r'$\mathbf{m_{\rm{DESI}}}$= %.2f' % p1_depth)
                        # 90% > than requirement
                        q10= np.percentile(self.decam.get(key)[keep], q=10)
                        #ax[row,col].axvline(q10,
                        #                c=band2color(band),ls='dotted',lw=2,
                        #                label='q10= %.2f' % q10)
                        mybins= mybins[:-1]
                        lasth= myh[mybins <= q10][-1]
                        myh= np.append(myh[mybins <= q10], lasth)
                        mybins= np.append(mybins[mybins <= q10], q10)
                        ax[row,col].fill_between(mybins,[0]*len(myh),myh,
                                                 where= mybins <= q10, interpolate=True,step='post',
                                                 color=band2color(band),alpha=0.5)
            # mosaic
            if self.mosaic:
                row=d_row['mosaic']
                col=d_col[which]
                for band in set(self.mosaic.filter):
                    mos_color='k'
                    keep= self.mosaic.filter == band
                    myh, mybins= myhist_step(ax[row,col],self.mosaic.get(key)[keep], bins=bins,normed=True,
                                color=band2color(band),ls='solid',lw=2,
                                label='%s (Mosaic3)' % band, return_vals=True)
                    # requirements
                    if which == 'gal':
                        p1_depth= depth_obj.get_single_pass_depth(band,which,'mosaic')
                        ax[row,col].axvline(p1_depth,
                                        c=band2color(band),ls='dashed',lw=1,
                                        label=r'$\mathbf{m_{\rm{DESI}}}$= %.2f' % p1_depth)
                        # 90% > than requirement
                        q10= np.percentile(self.mosaic.get(key)[keep], q=10)
                        #ax[row].axvline(q10,
                        #                c=mos_color,ls='dotted',lw=2,
                        #                label='q10= %.2f' % q10)
                        mybins= mybins[:-1]
                        lasth= myh[mybins <= q10][-1]
                        myh= np.append(myh[mybins <= q10], lasth)
                        mybins= np.append(mybins[mybins <= q10], q10)
                        ax[row,col].fill_between(mybins,[0]*len(myh),myh,
                                                 where= mybins <= q10, interpolate=True,step='post',
                                                 color=band2color(band),alpha=0.5)
        # Label
        cam={}
        cam['0']='DECaLS'
        cam['1']='MzLS'
        for row in [1,0]:
            for col in [0,1]:
                if legend:
                    leg=ax[row,col].legend(loc=(0.,1.02),ncol=3,fontsize=FS-5)
                ax[row,col].tick_params(axis='both', labelsize=tickFS)
                if ylim:
                    ax[row,col].set_ylim(ylim)
                if xlim:
                    ax[row,col].set_xlim(xlim)
            ylab=ax[row,0].set_ylabel('PDF (%s)' % cam[str(row)],fontsize=FS)
        for col,which in zip([0,1],['psf','gal']): 
            key= which+'depth_extcorr'
            xlab=ax[1,col].set_xlabel(r'%s' % col2plotname(key),fontsize=FS) 
        # Hide axis labels
        for col in [0,1]: 
            ax[0,col].set_xticklabels([])
        for row in [0,1]:
            ax[row,1].set_yticklabels([])
        savefn='hist_depth.png'
        bbox=[xlab,ylab]
        if legend:
            bbox.append(leg)
        plt.savefig(savefn, bbox_extra_artists=bbox, bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 


    def get_lim(self,col):
        d= dict(rarms= (0,0.5),
                decrms= (0,0.5),
                phrms= (0,0.5),
                raoff= (-0.7,0.7),
                decoff= (-0.7,0.7),
                phoff= (-6,6),
                err_on_phoff= (0,0.08),
                err_on_radecoff= (0,0.3))
        return d.get(col,None)

    def plot_2d_scatter(self,prefix=''):
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        xy_sets= [('rarms','decrms'),
                  ('raoff','decoff'),
                  ('phrms','phoff')]
        for x_key,y_key in xy_sets:
            fig,ax= plt.subplots(2,1,figsize=(8,10))
            plt.subplots_adjust(hspace=0.2,wspace=0)
            # decam
            if self.decam:            
                row=0
                x,y= self.decam.get(x_key), self.decam.get(y_key)
                for band in set(self.decam.filter):
                    keep= self.decam.filter == band
                    myscatter(ax[row],x[keep],y[keep], 
                              color=band2color(band),s=10.,alpha=0.75,
                              label='%s (DECam, %d)' % (band,self.num_exp['decam_'+band]))
            # mosaic
            if self.mosaic:            
                row=1
                x,y= self.mosaic.get(x_key), self.mosaic.get(y_key)
                for band in set(self.mosaic.filter):
                    keep= self.mosaic.filter == band
                    myscatter(ax[row],x[keep],y[keep], 
                              color=band2color(band),s=10.,alpha=0.75,
                              label='%s (Mosaic3, %d)' % (band,self.num_exp['mosaic_'+band]))
            # Crosshairs
            for row in range(2):
                ax[row].axhline(0,c='k',ls='dashed',lw=1)
                ax[row].axvline(0,c='k',ls='dashed',lw=1)
            # Label
            xlab=ax[1].set_xlabel(col2plotname(x_key),fontsize=FS) #0.45'' galaxy
            for row in range(2):
                ylab=ax[row].set_ylabel(col2plotname(y_key),fontsize=FS)
                ax[row].tick_params(axis='both', labelsize=tickFS)
                leg=ax[row].legend(loc='upper right',scatterpoints=1,markerscale=3.,fontsize=FS)
                if self.get_lim(x_key):
                    ax[row].set_xlim(self.get_lim(x_key))
                if self.get_lim(y_key):
                    ax[row].set_ylim(self.get_lim(y_key))
            savefn='rms_2panel_%s_%s.png' % (x_key,y_key)
            plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
            plt.close() 
            print "wrote %s" % savefn 

    def plot_astro_photo_scatter(self,prefix=''):
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        x_key,y_key= 'err_on_radecoff','err_on_phoff'
        fig,ax= plt.subplots(2,1,figsize=(8,10))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        # g,r band
        row=0
        if self.decam:            
            x,y= self.decam.get(x_key), self.decam.get(y_key)
            for band in ['g','r']:
                keep= self.decam.filter == band
                if np.where(keep)[0].size > 0:
                    myscatter(ax[row],x[keep],y[keep], 
                              color=band2color(band),s=10.,alpha=0.75,
                              label='%s (DECam, %d)' % (band,len(x[keep])))
        # z-band
        row=1
        if self.decam:            
            x,y= self.decam.get(x_key), self.decam.get(y_key)
            for band in ['z']:
                keep= self.decam.filter == band
                if np.where(keep)[0].size > 0:
                    myscatter(ax[row],x[keep],y[keep], 
                              color=band2color(band),s=10.,alpha=0.75,
                              label='%s (DECam, %d)' % (band,len(x[keep])))
        if self.mosaic:            
            x,y= self.mosaic.get(x_key), self.mosaic.get(y_key)
            for band in set(self.mosaic.filter):
                keep= self.mosaic.filter == band
                if np.where(keep)[0].size > 0:
                    myscatter(ax[row],x[keep],y[keep], 
                              color='k',s=10.,alpha=0.75,
                              label='%s (Mosaic3, %d)' % (band,len(x[keep])))
        # Crosshairs
        ax[0].axhline(0.01,c='k',ls='dashed',lw=1)
        ax[1].axhline(0.02,c='k',ls='dashed',lw=1)
        for row in range(2):
            ax[row].axvline(0.03,c='k',ls='dashed',lw=1)
        # Label
        xlab=ax[1].set_xlabel(col2plotname(x_key),fontsize=FS) #0.45'' galaxy
        for row in range(2):
            ylab=ax[row].set_ylabel(col2plotname(y_key),fontsize=FS)
            ax[row].tick_params(axis='both', labelsize=tickFS)
            leg=ax[row].legend(loc='upper right',scatterpoints=1,markerscale=3.,fontsize=FS)
            if self.get_lim(x_key):
                ax[row].set_xlim(self.get_lim(x_key))
            if self.get_lim(y_key):
                ax[row].set_ylim(self.get_lim(y_key))
        savefn='astro_photo_error.png'
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def col2name(self,col):
        d=dict(skycounts='Sky Level',
               skyrms='Sky RMS',
               skymag='Sky Brightness',
               zpt='Zeropoint',
               seeing='Seeing',
               airmass='Airmass',
               transp='Atmospheric Transp',
               phoff='Photometric Offset',
               raoff='Ra Offset',
               decoff='Dec Offset')
        return d[col]

    def print_ccds_table(self):
        print('CCDS TABLE')
        print('statistic & units & DECam g & DECam r & DECam z & Mzls z')
        for col in ['skycounts','skyrms','skymag','zpt','seeing','airmass','transp','phoff','raoff','decoff']:
            if col in ['phoff','raoff','decoff']:
                # has an uncertainty
                # Assume we have decam
                text= '& %s & units & %.2f(%.2f) & %.2f(%.2f) & %.2f(%.2f)' % \
                        (self.col2name(col),
                             np.median( self.decam.get(col)[self.decam.filter == 'g'] ),
                                np.std( self.decam.get(col)[self.decam.filter == 'g'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'r'] ),
                                np.std( self.decam.get(col)[self.decam.filter == 'r'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'z'] ),
                                np.std( self.decam.get(col)[self.decam.filter == 'z'] )
                        )
                if self.mosaic:
                    text= text + ' & %.2f(%.2f) \\\\' % \
                            (np.median( self.mosaic.get(col)[self.mosaic.filter == 'z'] ),
                                np.std( self.mosaic.get(col)[self.mosaic.filter == 'z'] )
                            )
                else:
                    text= text + ' & --(--) \\\\' 
                print(text)
            else:
                # No error computed
                # Assume we have decam
                text= '& %s & units & %.2f & %.2f & %.2f' % \
                        (self.col2name(col),
                             np.median( self.decam.get(col)[self.decam.filter == 'g'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'r'] ),
                             np.median( self.decam.get(col)[self.decam.filter == 'z'] )
                        )
                if self.mosaic:
                    text= text +  ' & %.2f  \\\\' % \
                            (np.median( self.mosaic.get(col)[self.mosaic.filter == 'z'] ),
                            )
                else:
                    text= text +  ' & --  \\\\'  
                print(text)

    def print_requirements_table(self):
        print('DESI REQUIREMENTS TABLE')
        print('camera & band & psfdepth: q10 q50 desi_req & galdepth: q10 q50 desi_req')
        phrms= dict(g=0.01,r=0.01,z=0.02)
        depth_obj= DepthRequirements()
        need_cols= ['psfdepth_extcorr','galdepth_extcorr',
                    'err_on_radecoff','err_on_phoff']
        for band in ['g','r','z']:
            keep= (self.decam.filter == band)
            for col in need_cols:
                keep *= (np.isfinite(self.decam.get(col)))
            p1_depth_psf= depth_obj.get_single_pass_depth(band,'psf','decam')
            p1_depth_gal= depth_obj.get_single_pass_depth(band,'gal','decam')
            print('DECam & %s & %.2f | %.2f | %.2f & %.2f | %.2f | %.2f \\\\' % \
                    (band,np.percentile( self.decam.psfdepth_extcorr[keep], q=10),
                          np.percentile( self.decam.psfdepth_extcorr[keep], q=50),
                            p1_depth_psf,
                          np.percentile( self.decam.galdepth_extcorr[keep], q=10),
                          np.percentile( self.decam.galdepth_extcorr[keep], q=50),
                            p1_depth_gal,
                          #np.median( self.decam.err_on_radecoff[keep]),
                          #  0.030,
                          #np.median( self.decam.err_on_phoff[keep]),
                          #  phrms[band],
                    ))
        if self.mosaic:
            for band in ['z']:
                keep= self.mosaic.filter == band
                for col in need_cols:
                    keep *= (np.isfinite(self.mosaic.get(col)))
                p1_depth_psf= depth_obj.get_single_pass_depth(band,'psf','mosaic')
                p1_depth_gal= depth_obj.get_single_pass_depth(band,'gal','mosaic')
                print('MOSAIC-3 & %s & %.2f | %.2f | %.2f & %.2f | %.2f | %.2f \\\\' % \
                        (band,np.percentile( self.mosaic.psfdepth_extcorr[keep], q=10),
                              np.percentile( self.mosaic.psfdepth_extcorr[keep], q=50),
                                p1_depth_psf,
                              np.percentile( self.mosaic.galdepth_extcorr[keep], q=10),
                              np.percentile( self.mosaic.galdepth_extcorr[keep], q=50),
                                p1_depth_gal,
                              #np.median( self.mosaic.err_on_radecoff[keep]),
                              #  0.030,
                              #np.median( self.mosaic.err_on_phoff[keep]),
                              #  phrms[band],
                        ))





#######
# Needed exposure time calculator
#######
# compare to obsdb's value
# zp= ZptsTneed(zpt_fn='/global/cscratch1/sd/kaylanb/publications/observing_paper/neff_empirical/test_1000-zpt.fits.gz',camera=camera)
# db= LoadDB(camera=camera) 
# Match
# m= db.match_to_zeropoints(tab=zp.data)

class CalculateTneed(object):
    def calc(self,camera,filter_arr,fwhm_pix,transp,airmass,ebv,msky_ab_per_arcsec2):
        self.fid= params.get_fiducial(camera=camera)
        # Obsbot uses Neff_gal not psf
        neff_fid = self.Neff_gal(self.fid.fwhm0 / self.fid.pixscale)
        neff     = self.Neff_gal(fwhm_pix)
        
        Kco_arr= np.zeros(len(transp))-1
        Aco_arr= np.zeros(len(transp))-1
        sky0_arr= np.zeros(len(transp))-1
        t0_arr= np.zeros(len(transp))-1
        for band in self.fid.bands:
            hasband= filter_arr == band
            Kco_arr[hasband]= self.fid.Kco[band]
            Aco_arr[hasband]= self.fid.Aco[band]
            sky0_arr[hasband]= self.fid.sky0[band]
            t0_arr[hasband]= self.fid.t0[band]
        #
        scaling =  1./transp**2 *\
                   10.**(0.8 * Kco_arr * (airmass - 1.)) *\
                   10.**(0.8 * Aco_arr * ebv) *\
                   (neff / neff_fid) *\
                   10.**(-0.4 * (msky_ab_per_arcsec2 - sky0_arr))
        assert(np.all(scaling > 0.))
        return scaling * t0_arr

    def Neff_gal(self,fwhm_pix):
        r_half = 0.45 #arcsec
        # magic 2.35: convert seeing FWHM into sigmas in arcsec.
        return 4. * np.pi * (fwhm_pix / 2.35)**2 +\
               8.91 * r_half**2 +\
               self.fid.pixscale**2/12.



class ZptsTneed(object): 
    '''depths: annotated ccd vs. legacy'''
    def __init__(self,zpt_fn=None,camera=None):
        assert(camera in ['decam','mosaic'])
        self.camera = camera
        self.data= fits_table(zpt_fn)
        # process
        self.apply_cuts()
        #self.convert_units()
        self.add_cols()

    def convert_units(self):
        if self.camera == 'decam':
            self.data.set('transp', self.data.transp * 10**(-0.4* 2.5*np.log10(self.data.gain)))
            self.data.set('skymag', self.data.skymag + 2.5*np.log10(self.data.gain))
        elif self.camera == 'mosaic':
            self.data.set('transp', self.data.transp * 10**(-0.4* 2.5*np.log10(self.data.gain)))
            self.data.set('skymag', self.data.skymag + 2.5*np.log10(self.data.gain))
    
    def add_cols(self):
        # Add EBV
        sfd = SFDMap()
        self.data.set('ebv', sfd.ebv(self.data.ra, self.data.dec) )
        # Add tneed 
        kwargs= dict(camera= self.camera,
                     filter_arr= self.data.filter,
                     fwhm_pix= self.data.fwhm,
                     transp= self.data.transp,
                     airmass= self.data.airmass,
                     ebv= self.data.ebv,
                     msky_ab_per_arcsec2= self.data.skymag)
        self.data.set('tneed', CalculateTneed().calc(**kwargs) )
 
    def apply_cuts(self):
        self.zpt_succeeded()
        #self.remove_duplicate_expnum()

    def zpt_succeeded(self):
        keep= (np.isfinite(self.data.zpt))*\
              (self.data.zpt > 0)*\
              (self.data.fwhm > 0)
        print('zpt_succeeded: keeping %d/%d' % \
               (np.where(keep)[0].size,len(keep)))
        self.data.cut(keep)

    def remove_duplicate_expnum(self):
        # Keep everything unless told otherwise
        keep= np.ones(len(self.data),bool)
        # Loop over exposures
        num_dup_exp=0
        num_dup_ccds=0
        expnums= set(self.data.expnum)
        print('Removing duplicates')
        for cnt,expnum in enumerate(expnums):
            if cnt % 100 == 0: 
                print('%d/%d' % (cnt,len(expnums)))
            exp_inds= self.data.expnum == expnum
            if self.camera == 'decam':
                max_sz= 62
            elif self.camera == 'mosaic':
                max_sz= 4
            # Have duplicates?
            if len(self.data[exp_inds]) > max_sz:
                print('Found duplicate expnum=%s' % expnum)
                num_dup_exp += 1
                for ccdname in set(self.data.ccdname[exp_inds]):
                    inds= np.where( (exp_inds)*\
                                    (self.data.ccdname == ccdname) )[0]
                    # We can delete at least one of these
                    if inds.size > 1:
                        num_dup_ccds += inds.size-1
                        keep[ inds[1:] ] = False
        print('number duplicate exposures=%d, ccds=%d' % (num_dup_exp,num_dup_ccds))
        self.data.cut(keep)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                            description='Generate a legacypipe-compatible CCDs file \
                                        from a set of reduced imaging.')
    parser.add_argument('--leg_dir',action='store',default='/global/cscratch1/sd/kaylanb/kaylan_Test',required=False)
    args = parser.parse_args()

    fns= {}
    fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-decam44k.fits.gz'
    fns['mosaic_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-mosaic42k.fits.gz'
    a= ZeropointHistograms(decam_zpts=fns['decam_l'],
                           mosaic_zpts=fns['mosaic_l'])
