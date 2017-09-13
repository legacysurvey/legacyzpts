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

from legacyzpts.qa.paper_plots import Depth
from legacyzpts.qa.paper_plots import myscatter
from legacyzpts.qa.params import band2color,col2plotname,getrms

mygray='0.6'

#######
# Scatter of psf and galdepth from annotated-ccds vs. legacy zeropoints estimations for depth
#######

class AnnotatedVsLegacy(object): 
    '''depths: annotated ccd vs. legacy'''
    def __init__(self,annot_ccds=None,legacy_zpts=None,camera=None):
        assert(camera in ['decam','mosaic'])
        self.camera= camera
        self.annot= annot_ccds
        #
        self.legacy= legacy_zpts
        if self.annot:
            self.annot= fits_table(self.annot)
        if self.legacy:
            self.legacy= fits_table(self.legacy)
        self.add_keys()
        self.match()
        self.apply_cuts()
        # Plot
        if 'psfdepth' in self.legacy.get_columns():
            self.plot_scatter()
   
    def match(self):
        '''
        ra,dec_center -- annot ccd center
        ra,dec -- legacy center
        '''
        m1, m2, d12 = match_radec(self.legacy.ra,self.legacy.dec, 
                                  self.annot.ra_center, self.annot.dec_center, 
                                  1./3600,nearest=True)
        self.legacy= self.legacy[m1]
        self.annot= self.annot[m2]
        # remove mismatching image_filenames
        print('looping over, %d' % len(self.legacy))
        legacy_base= [os.path.basename(fn) for fn in np.char.strip(self.legacy.image_filename)]
        annot_base= [os.path.basename(fn) for fn in np.char.strip(self.annot.image_filename)]
        print('looping finished')
        same= np.array(legacy_base) == \
              np.array(annot_base)
        self.legacy.cut(same)
        self.annot.cut(same)
        print('removing %d ccds b/c name mismatch, have this many left %d' % \
                (np.where(same == False)[0].size,len(self.legacy)))

    def apply_cuts(self):
        keep= np.ones(len(self.annot),bool)
        print('Before cuts= %d' % len(self.annot))
        # Annotated cuts
        if self.camera == 'decam':
            keep *= (self.annot.photometric)
        elif self.camera == 'mosaic':
            keep *= (self.annot.photometric)*\
                    (self.annot.bitmask == 0) 
        # Remove nans
        keep *= (self.annot.psfnorm_mean > 0)*\
                (self.annot.galnorm_mean > 0)* \
                (np.isfinite(self.annot.psfnorm_mean))*\
                (np.isfinite(self.annot.galnorm_mean))*\
                (np.isfinite(self.legacy.fwhm))
        # Cut on both
        self.annot.cut(keep)
        self.legacy.cut(keep)
        print('After cuts= %d' % len(self.annot))

    def add_keys(self):
        self.add_legacy_keys()
        self.add_annot_keys()
    
    def add_legacy_keys(self):
        if not 'gain' in self.legacy.get_columns():
            print('WARNING: cannot compute depths, not enough info in table')
        else:
            depth_obj= Depth(self.camera,
                             self.legacy.skyrms,self.legacy.gain,
                             self.legacy.fwhm,self.legacy.zpt)
            self.legacy.set('psfdepth', depth_obj.get_depth_legacy_zpts('psf'))
            self.legacy.set('galdepth', depth_obj.get_depth_legacy_zpts('gal'))

    def add_annot_keys(self):
        for key in ['psf','gal']:
            self.annot.set(key+'depth_2', self.get_true_depth(key))

    def get_true_depth(self,which='gal'):
        '''exactly agrees with listed galdepth or psfdepth value'''
        sigma= self.sigma_annot_ccds(self.annot.sig1,self.annot.ccdzpt,
                                     self.get_norm(which=which)) #ADU / sec
        return -2.5*np.log10(5 * sigma) + self.annot.ccdzpt # ADU / sec
    
    def get_norm(self,which=None):
        assert(which in ['gal','psf'])
        if which == 'psf':
            return self.annot.psfnorm_mean
        elif which == 'gal':
            return self.annot.galnorm_mean
  
    def sig1_to_ADU_per_sec(self,sig1,ccdzpt):
        '''annotated CCD sig1 for DECam
        this zeropointToScale() func converts nanomaggies to natual cameras system
        which for legacpipe.decam.py is ADU/sec'''
        return sig1 * NanoMaggies.zeropointToScale(ccdzpt) 

    def sigma_annot_ccds(self,sig1,ccdzpt,norm):
        '''norm is one of: {psf,gal}norm, {psf,gal}norm_mean'''
        return self.sig1_to_ADU_per_sec(sig1,ccdzpt) / norm

    def plot_hdu_diff(self):
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(figsize=(7,5))
        colors=['g','r','m']
        x= self.annot.image_hdu
        y= self.legacy.image_hdu - x
        is_wrong= np.where(y != 0)[0]
        print('matched ccds=%d, correct=%d, wrong=%d' % 
            (len(self.legacy),np.where(y == 0)[0].size,is_wrong.size))
        print('first 10 wrong hdu images:')
        for fn in list(set(np.char.strip(self.legacy.image_filename[is_wrong])))[:10]:
            print('%s' % fn)
        for band,color in zip(set(self.legacy.filter),colors):
            keep = self.legacy.filter == band
            myscatter(ax,x[keep],y[keep], 
                      color=color,m='o',s=10.,alpha=0.75,label='%s (%d)' % (band,len(x[keep])))
        # Legend
        ax.legend(loc=0,fontsize=FS-2)
        # Label
        xlab=ax.set_xlabel('image_hdu (Annotated CCDs)',fontsize=FS) 
        ylab=ax.set_ylabel('Legacy - Annot (image_hdu)',fontsize=FS) 
        ax.tick_params(axis='both', labelsize=tickFS)
        savefn='hdu_diff_%s.png' % self.camera
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def plot_scatter(self):
        # All keys and any ylims to use
        xlim= (19,25)
        ylim= (19,25)
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(2,1,figsize=(8,10))
        plt.subplots_adjust(hspace=0.2,wspace=0)
        for row,which in zip([0,1],['psf','gal']):
            col= which+'depth'
            for band in set(self.legacy.filter):
                keep= self.legacy.filter == band
                myscatter(ax[row],self.annot.get(col)[keep],self.legacy.get(col)[keep], 
                          color=band2color(band),m='o',s=10.,alpha=0.75,
                          label=band)
            ax[row].plot(ax[row].get_xlim(),ax[row].get_xlim(),c='k',ls='--',lw=2)
            # Text
            ax[row].text(0.05,0.9,r'%s' % col2plotname(col),
                         fontsize=FS-2,transform=ax[row].transAxes)
            diff= self.annot.get(col) - self.legacy.get(col) 
            rms= getrms(diff)
            q75_25= np.percentile(diff,q=75) - np.percentile(diff,q=25)
            ax[row].text(0.05,0.8,'rms=%.2f, q75-25=%.2f' % (rms,q75_25),
                         fontsize=FS-2,transform=ax[row].transAxes)
            # Legend
            leg=ax[row].legend(loc=(1.01,0.1),ncol=1,fontsize=FS-2)
            # Label
            xlab=ax[row].set_xlabel('Annotated CCDs',fontsize=FS) 
            ylab=ax[row].set_ylabel('Legacy Zeropoints',fontsize=FS) 
            ax[row].tick_params(axis='both', labelsize=tickFS)
            if ylim:
                ax[row].set_ylim(ylim)
            if xlim:
                ax[row].set_xlim(xlim)
        savefn='annot_legacy_scatter_%s.png' % self.camera
        plt.savefig(savefn, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def plot_scatter_2cameras(self,dec_legacy,dec_annot,
                                   mos_legacy,mos_annot):
        '''same as plot_scatter() but
        dec_legacy,dec_annot are the decam self.legacy and self.annot tables
        mos_legacy,mos_annot are the mosaic...
        '''
        # All keys and any ylims to use
        xlim= (20,28)
        ylim= (20,25)
        # Plot
        FS=14
        eFS=FS+5
        tickFS=FS
        fig,ax= plt.subplots(figsize=(7,5))
        colors=['m','k','g','r']
        offsets=[0,1,2,3]
        i=-1
        for camera in ['mosaic','decam']:
            for which in ['psf','gal']:
                i+=1
                color= colors[i]
                offset= offsets[i]
                col= which+'depth'
                if camera == 'decam':
                    x= dec_annot.get(col)
                    y= dec_legacy.get(col)
                elif camera == 'mosaic':
                    x= mos_annot.get(col)
                    y= mos_legacy.get(col)
                rms= getrms(x-y)
                myscatter(ax,x + offset,y, 
                          color=color,m='o',s=10.,alpha=0.75,
                          label='%s,%s: RMS=%.2f' % (camera,which,rms))
                ax.plot(np.array(xlim)+ offset,xlim,c=color,ls='--',lw=2)
        # Legend
        leg=ax.legend(loc=(0.,1.02),ncol=2,markerscale=3,fontsize=FS-2)
        # Label
        xlab=ax.set_xlabel('Depth (Annotated CCDs) + Offset',fontsize=FS) 
        ylab=ax.set_ylabel('Depth (Legacy Zeropoints)',fontsize=FS) 
        ax.tick_params(axis='both', labelsize=tickFS)
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
        savefn='annot_legacy_scatter_all.png'
        plt.savefig(savefn, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                            description='Generate a legacypipe-compatible CCDs file \
                                        from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--leg_dir',action='store',default='/global/cscratch1/sd/kaylanb/kaylan_Test',required=False)
    args = parser.parse_args()

    ####
    # Histograms and annotated ccd depth vs. zpt depth
    fns={}
    fns['decam_a']= '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/ccds-annotated-decals.fits.gz'
    fns['mosaic_a']= '/global/project/projectdirs/cosmo/data/legacysurvey/dr4/ccds-annotated-mzls.fits.gz'
    #fns['decam_l']= '/global/cscratch1/sd/kaylanb/publications/observing_paper/neff_empirical/test_1000-zpt.fits.gz'
    fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-decam44k.fits.gz'
    #fns['mosaic_l']= '/global/cscratch1/sd/kaylanb/publications/observing_paper/neff_empirical/mosaic-zpt.fits.gz'
    fns['mosaic_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-mosaic42k.fits.gz'
    
    a={}
    for camera in ['decam','mosaic']:
        a[camera]= AnnotatedVsLegacy(annot_ccds=fns['%s_a' % camera], 
                                     legacy_zpts=fns['%s_l' % camera],
                                     camera='%s' % camera)
    a[camera].plot_scatter_2cameras(dec_legacy=a['decam'].legacy, dec_annot=a['decam'].annot,
                                    mos_legacy=a['mosaic'].legacy, mos_annot=a['mosaic'].annot)
 
    ######## 
    ## TEST image_hdu is correct
    camera='decam'
    #fns['decam_l']= '/global/cscratch1/sd/kaylanb/dr5_zpts/survey-ccds-25bricks-hdufixed.fits.gz'
    #fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey_ccds_lp25_hdufix2.fits.gz'
    #fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-25bricks-fixed.fits.gz'
    fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/legacypipe_decam_all.fits.gz'
    #fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/legacypipe_decam2k.fits.gz'
    obj= AnnotatedVsLegacy(annot_ccds=fns['%s_a' % camera], 
                         legacy_zpts=fns['%s_l' % camera],
                         camera='%s' % camera)
    obj.plot_hdu_diff()
