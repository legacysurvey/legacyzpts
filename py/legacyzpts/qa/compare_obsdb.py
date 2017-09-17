'''
TO RUN:
idl vs. legacy comparison: python -c "from simulate_survey import Legacy_vs_IDL;a=Legacy_vs_IDL(camera='decam',leg_dir='/global/cscratch1/sd/kaylanb/kaylan_Test',idl_dir='/global/cscratch1/sd/kaylanb/arjundey_Test')"
idl vs. legacy number star matches: python -c "from simulate_survey import sn_not_matched_by_arjun;sn_not_matched_by_arjun('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-*-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')"
idl vs. legacy number star matches: python -c "from simulate_survey import number_matches_by_cut;number_matches_by_cut('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-*-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')"
oplot stars on ccd: python -c "from simulate_survey import run_imshow_stars;run_imshow_stars('/global/cscratch1/sd/kaylanb/kaylan_Test/decam/DECam_CP/CP20170326/c4d_170327_042342_oki_r_v1-35-extra.pkl',arjun_fn='/global/cscratch1/sd/kaylanb/arjundey_Test/matches-c4d_170327_042342_oki_r_v1.fits')"
'''

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
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

from legacyzpts.qa.params import get_fiducial, band2color, col2plotname
from legacyzpts.common import fits2pandas,save_png
from legacyzpts.qa.paper_plots import myerrorbar
mygray='0.6'

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
        self.fid= get_fiducial(camera=camera)
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

 
class LoadDB(object):
    def __init__(self,camera=None):
        assert(camera in ['decam','mosaic'])
        self.camera= camera
        self.fid= get_fiducial(self.camera)
        self.data= self.load()
        self.apply_cuts()
        self.expf_to_tneed()

    def load(self):
        import sqlite3
        self.db_dir= os.path.join(os.getenv('CSCRATCH'),'zeropoints/obsbot/obsdb')
        if self.camera == 'decam':
            fn= os.path.join(self.db_dir,'decam.sqlite3')
        elif self.camera == 'mosaic':
            fn= os.path.join(self.db_dir,'mosaic3.sqlite3')
        print('Reading sqlite db: %s' % fn)
        conn = sqlite3.connect(fn)
        c= conn.cursor() 
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")   
        print('Has tables: ',c.fetchall())        
        c.execute("select * from obsdb_measuredccd")
        cols= [col[0] for col in c.description] 
        print('Reading table obsdb_measuredccd, cols are:',cols) 
        # List of tuples, length = number exposures
        a=c.fetchall()
        # List of tuples, length = number cols
        b=zip(*a) 
        data=fits_table()   
        for col in cols:
            data.set(col, np.array( b[cols.index(col)]) )
        return data

    def apply_cuts(self): 
        # Remove whitespaces
        self.data.set('band', self.data.get('band').astype(np.string_))
        self.data.set('band', np.char.strip(self.data.band))
        print('Initially %d' % len(self.data))
        # obstype: {u'dark', u'dome flat', u'object', u'zero'}
        # band: {u'', u'VR', u'Y', u'g', u'i', u'r', u'solid', u'u', u'z'}
        keep= (self.data.obstype == 'object')
        if self.camera == 'decam':
            keep *= (np.any([self.data.band == 'g',
                             self.data.band == 'r',
                             self.data.band == 'z'],axis=0) )
        elif self.camera == 'mosaic':
            keep *= (self.data.band == 'zd')
        self.data.cut(keep)
        print('Object and band cuts %d' % len(self.data))
        # rename_bands:
        if self.camera == 'mosaic':
            assert( np.all(self.data.band == 'zd') )
            self.data.set('band', np.array(['z']*len(self.data)) )
    
    def expf_to_tneed(self):
        tneed= self.data.expfactor
        for band in set(self.data.band):
            keep= self.data.band == band 
            tneed[keep] *= self.fid.t0[band] 
        #self.data.set('tneed', tneed / 1.1) # expfactor is 1.1x what it should be
        self.data.set('tneed', tneed)

    def match_to_zeropoints(self,tab):
        '''tab -- fits_table'''
        m= defaultdict(list) 
        expnums= list( set(tab.expnum).intersection(set(self.data.expnum)) )
        print('%d matching expnums' % len(expnums))
        for expnum in expnums[:1000]:
            i_zp= tab.expnum == expnum
            i_db= self.data.expnum == expnum
            m['zp_band'].append( tab.filter[i_zp][0] )
            m['db_band'].append( self.data.band[i_db][0] )
            assert(m['zp_band'][-1] == m['db_band'][-1])
            m['zp_tneed_med'].append( np.median(tab.tneed[i_zp]) )
            m['zp_tneed_std'].append( np.std(tab.tneed[i_zp]) )
            m['db_expf'].append( self.data.expfactor[i_db][0] )
            m['db_tneed'].append( self.data.tneed[i_db][0] )
            m['zp_gain'].append( np.median(tab.gain[i_zp]) )
            m['zp_mjd'].append( tab.mjd_obs[i_zp][0] )
            for key,dbkey in zip(['fwhm','skymag','transp','zpt'],
                                 ['seeing','sky','transparency','zeropoint']):
                m['zp_'+key].append( np.median(tab.get(key)[i_zp]) ) 
                m['db_'+dbkey].append( self.data.get(dbkey)[i_db][0] )
        data= fits_table()
        for key in m.keys():
            data.set(key, np.array(m.get(key))) 
        return data

    def make_tneed_plot(self,decam,mosaic=None):
        '''makes legacy zpts vs. db tneed plots
        decam -- return fits_table from match_zeropoints above (rquired)
        mosaic -- ditto (optional)
        '''
        # Plot
        xlim=None
        ylim=(-100,100)
        FS=14
        eFS=FS+5
        tickFS=FS
        # assume decam_tab always provided
        if mosaic:
            fig,ax= plt.subplots(4,1,figsize=(4,12))
        else:
            fig,ax= plt.subplots(3,1,figsize=(4,9))
        # DECam grz Diff
        resid= decam.zp_tneed_med - decam.db_tneed
        for row,band in zip([0,1,2],['g','r','z']):
            keep= decam.zp_band == band
            if np.where(keep)[0].size > 0:
                myerrorbar(ax[row],decam.zp_mjd[keep],resid[keep], 
                           yerr=decam.zp_tneed_std[keep],
                           color=band2color(band),m='o',s=10.,label='%s (DECam)' % band)
        # Mosaic z Diff
        if mosaic:
            resid= mosaic.zp_tneed_med - mosaic.db_tneed
            for row,band in zip([3],['z']):
                keep= mosaic.zp_band == band
                if np.where(keep)[0].size > 0:
                    myerrorbar(ax[row],mosaic.zp_mjd[keep],resid[keep], 
                               yerr=mosaic.zp_tneed_std[keep],
                               color=band2color(band),m='o',s=10.,label='%s (Mosaic3)' % band)
        # Label
        if mosaic:
            nplots=4
        else:
            nplots=3
        xlab=ax[nplots-1].set_xlabel('MJD',fontsize=FS)
        ylab=ax[nplots-1].set_ylabel('tneed (Legacy - Obsbot DB)',fontsize=FS)
        for row in range(nplots):
            ax[row].tick_params(axis='both', labelsize=tickFS)
            leg=ax[row].legend(loc='upper left',scatterpoints=1,markerscale=1.,fontsize=FS)
            if xlim:
                ax[row].set_xlim(xlim)
            if ylim:
                ax[row].set_ylim(ylim)
        savefn='tneed_plot.png' 
        plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
        plt.close() 
        print "wrote %s" % savefn 

    def scatter_plots(self,data,camera='decam'):
        """data is fits_table of legacy obsdb matched values"""
        dat= fits2pandas(data)
        val_cnts= dat['db_band'].value_counts().to_frame()
        val_cnts.plot.bar()
        save_png('./','hist_bands_%s' % camera,tight=False)

        xlims=defaultdict(lambda: None)
        xlims['db_tneed']= (0,300)
        px_scale= 0.262
        dat['zp_seeing']= dat.loc[:,'zp_fwhm'] * px_scale
        xnames= ['db_tneed','db_zeropoint','db_sky',
                 'db_seeing','db_transparency']
        ynames= ['zp_tneed_med','zp_zpt','zp_skymag',
                 'zp_seeing','zp_transp']
        for xname,yname in zip(xnames,ynames):
            fig,ax= plt.subplots(3,1,figsize=(5,10))
            plt.subplots_adjust(hspace=0.2)
            for cnt,band in enumerate(['g','r','z']):
                isBand= dat['db_band'] == band
                if np.any(isBand):
                    dat[isBand].plot(ax=ax[cnt], kind='scatter', 
                                     x=xname,y=yname,
                                     label=band,
                                     xlim=xlims[xname],
                                     ylim=xlims[xname])
                    if xlims[xname]:
                      ax[cnt].plot(xlims[xname],xlims[xname],'k--')
                    save_png('./','scatter_%s_%s' % (xname,camera),tight=False)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                            description='Generate a legacypipe-compatible CCDs file \
                                        from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--leg_dir',action='store',default='/global/cscratch1/sd/kaylanb/kaylan_Test',required=False)
    args = parser.parse_args()
  
    fns={}
    fns['decam_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-decam44k.fits.gz'
    #fns['mosaic_l']= '/global/cscratch1/sd/kaylanb/publications/observing_paper/neff_empirical/mosaic-zpt.fits.gz'
    fns['mosaic_l']= '/global/cscratch1/sd/kaylanb/test/legacypipe/py/survey-ccds-mosaic42k.fits.gz'
    
    #######
    # Needed exposure time calculator
    # Compare Obsbot DB tneed to legacy zeropoints tneed
    #######
    m={}
    for camera in ['mosaic','decam']:
        zp= ZptsTneed(zpt_fn=fns['%s_l' % camera],camera=camera)
        db= LoadDB(camera=camera) 
        # Match
        m[camera]= db.match_to_zeropoints(tab=zp.data)
    #db.make_tneed_plot(decam=m['decam'])
    db.make_tneed_plot(decam=m['decam'],mosaic=m['mosaic'])
    db.scatter_plots(m['decam'],camera='decam')
    db.scatter_plots(m['mosaic'],camera='mosaic')


