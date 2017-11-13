"""
Run this script on the "*-zpts.fits" table for each camera to produce the 
    "survey-ccds*.fits" table for that camera. 

This script also makes the "*-legacypipe.fits" tables from the "survey-ccds*.fits" 
    tables, which have a fraction of "survey-ccds*.fits" info and slightly different
    column names but which are the *minimum* inputs to the legacypipe pipeline
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import numbers
import seaborn as sns

import fitsio
from legacyzpts.legacy_zeropoints import create_legacypipe_table
from legacyzpts.runmanager.qa import big2small_endian
from legacyzpts.common import dobash
# Sphinx build would crash
try:
    from astrometry.util.fits import fits_table, merge_tables
except ImportError:
    pass


CAMERAS= ['decam','90prime','mosaic']

# Bit codes for why a CCD got cut, used in cut_ccds().
CCD_CUT_BITS= dict(
    err_legacyzpts = 0x1,
    not_grz = 0x2,
    not_third_pix = 0x4, # Mosaic3 one-third-pixel interpolation problem
    exptime_lt_30 = 0x8,
    ccdnmatch_lt_20 = 0x10, 
    zpt_diff_avg = 0x20, 
    zpt_small = 0x40,  
    zpt_large = 0x80,
    sky_is_bright = 0x100,
    badexp_file = 0x200,
    )

def get_bits(name):
    return CCD_CUT_BITS[name]

class LegacyzptsCuts(object):
    """Applies legacyzpts cuts

    Args:
        T_zpt: '*-zpts.fits' table
    """

    def __init__(self,T_zpt,camera=None):
        assert(camera in CAMERAS)
        self.camera= camera
        self.T= T_zpt.copy()
        self.df= pd.DataFrame({'err':big2small_endian(self.T.err_message),
                               'band':big2small_endian(self.T.filter)})
        self.df['err']= self.df['err'].str.strip()

    def cuts(self,good=None,ccd_cuts=None):
        """Returns bool array, True where CCD should be kept"""
        if good is None:
            good= np.ones(len(self.T),bool)
        if ccd_cuts is None:
            ccd_cuts = np.zeros(len(self.T), np.int16)
        assert(len(good) == len(self.T))
        assert(len(ccd_cuts) == len(self.T))

        n0 = sum(good)
        for name in set(self.df.loc[self.df['err'].str.len() > 0,'err']):
            crit= self.df['err'] == name
            good[crit] = False
            ccd_cuts[crit] += get_bits('err_legacyzpts')
            #continue as usual
            n = sum(good)
            print('Flagged', n0-n, 'more:',
                  name)
            n0 = n

        for name,crit in [
            ("not_grz", (self.correct_bands())),
            #('not_third_pix', (self.third_pix())),
        ]:
            good[crit] = False
            ccd_cuts[crit] += get_bits(name)
            #continue as usual
            n = sum(good)
            print('Flagged', n0-n, 'more:',
                  name)
            n0 = n
        return good,ccd_cuts

    #def err_message(self):
    #    self.good[self.df['err'].str.len() > 0]= False

    def third_pix(self):
        if self.camera == 'mosaic':
            # The 1/3-pixel shift problem was fixed in hardware on MJD 57674,
            # so only check for problems in data before then.
            return ((self.T.has_yshift == False) & 
                    (self.T.mjd_obs < 57674.))

    def correct_bands(self):
        return self.df['band'].str.strip().isin(self.bands()) == False

    def bands(self):
        return {'decam':['g','r','z'],
                'mosaic':['z'],
                '90prime':['g','r']
               }[self.camera]

def get_badexp_fn(camera):
    url='https://desi.lbl.gov/svn/decam/code/'
    if camera == 'decam':
        url += 'observing/trunk/obstatus/bad_expid.txt'
    elif camera == 'mosaic':
        url += 'mosaic3/trunk/obstatus/bad_expid.txt'
    dobash('svn export %s %s_bad_expid.txt --force' % (url,camera))
    return '%s_bad_expid.txt' % camera


class LegacypipeCuts(object):
    """Applies legacyipe cuts

    Args: 
        T_leg: '*-zpt.fits' table
        camera:
        good: bool array to start with, if None then all True
    """
    def __init__(self,T_zpt,camera=None):
        assert(camera in CAMERAS)
        self.camera= camera
        self.ccds= create_legacypipe_table(T=T_zpt.copy(), camera=self.camera)
        # From legacypipe
        if self.camera == 'decam':
            self.ccds.ccdzpt += 2.5 * np.log10(self.ccds.exptime)

    def cuts(self, good=None, ccd_cuts=None):
        """Returns bool array and bitmask, True where CCD should be kept"""
        if good is None:
            good= np.ones(len(self.ccds),bool)
        if ccd_cuts is None:
            ccd_cuts = np.zeros(len(self.ccds), np.int16)
        assert(len(good) == len(self.ccds))
        assert(len(ccd_cuts) == len(self.ccds))

        z0 = self.nominal_zeropoints()
        z0 = np.array([z0[f[0]] for f in self.ccds.filter])
        n0 = sum(good)
        for name,crit in [
            ('exptime_lt_30', (self.ccds.exptime < 30)),
            ('ccdnmatch_lt_20', (self.ccds.ccdnmatch < 20)),
            ('zpt_diff_avg',
             (np.abs(self.ccds.zpt - self.ccds.ccdzpt) > 0.1)),
            ('zpt_small',
             (self.ccds.zpt < self.min_zeropoint(z0))),
            ('zpt_large',
             (self.ccds.zpt > self.max_zeropoint(z0))),
            ('sky_is_bright',
             (self.sky_too_bright())),
            ('badexp_file',
             (self.bad_exposure())),
        ]:
            good[crit] = False
            ccd_cuts[crit] += get_bits(name)
            #continue as usual
            n = sum(good)
            print('Flagged', n0-n, 'more:',
                  name)
            n0 = n
        return good,ccd_cuts

    def bad_exposure(self):
        if self.camera == '90prime':
            print('Skipping, badexp file does not exist for 90prime')
            return np.zeros(len(self.ccds),bool)
        else:
            fn= get_badexp_fn(self.camera)
            badexp=pd.read_csv(fn,sep='\s+',header=None,usecols=[0],names=['expnum'],comment='#')
            isBad= pd.Series(big2small_endian(self.ccds.expnum)).isin(badexp['expnum'])
            return isBad.values 

    def sky_too_bright(self):
        if self.camera == 'mosaic':
            return self.ccds.ccdskycounts >= 150
        else:
            return np.zeros(len(self.ccds),bool)
                
    def nominal_zeropoints(self):
        if self.camera == 'decam':
            return dict(g = 25.08,
                        r = 25.29,
                        z = 24.92)
        elif self.camera == 'mosaic':
            return dict(z = 26.20)
        elif self.camera == '90prime':
            return dict(g = 25.74,
                        r = 25.52,)

    def min_zeropoint(self,z0):
        """z0: nominal zeropoint"""
        if self.camera == 'decam':
            return z0 - 0.5	
        elif self.camera == 'mosaic':
            return z0 - 0.6
        elif self.camera == '90prime':
            return z0 - 0.5

    def max_zeropoint(self,z0):
        """z0: nominal zeropoint"""
        if self.camera == 'decam':
            return z0 + 0.25	
        elif self.camera == 'mosaic':
            return z0 + 0.6
        elif self.camera == '90prime':
             return z0 + 0.18

def write_survey_ccds(ccd_cuts,T_zpt,camera=None,gzip=True):
    assert(camera in CAMERAS)
    T_zpt.set('ccd_cuts',ccd_cuts)
    fn='survey-ccds-'+camera+'.fits'
    T_zpt.writeto(fn)
    print('Wrote %s' % fn)
    if gzip:
        dobash('gzip %s' % fn)
        print('Wrote '+fn+'.gz')
    T_leg= create_legacypipe_table(T=T_zpt, camera=camera)
    T_leg.set('ccd_cuts',ccd_cuts)
    fn=fn.replace('.fits','-legacypipe.fits')
    T_zpt.writeto(fn)
    print('Wrote %s' % fn)
    if gzip:
        dobash('gzip %s' % fn)
        print('Wrote '+fn+'.gz')
	

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--zpt_fn',action='store',required=True)
    args = parser.parse_args()

    T_zpt= fits_table(args.zpt_fn)
    # Empty filter strings mess up "for b in ccd.filter"
    isEmpty= T_zpt.filter == ' '
    if len(T_zpt[isEmpty]) > 0:
        print('Removed %d, filter is empty string' % len(T_zpt[isEmpty]))
        T_zpt.cut(isEmpty == False)

    Z= LegacyzptsCuts(T_zpt,args.camera)
    good,ccd_cuts= Z.cuts()

    L= LegacypipeCuts(T_zpt,args.camera)
    good,ccd_cuts= L.cuts(good=good,ccd_cuts=ccd_cuts)

    write_survey_ccds(ccd_cuts,T_zpt,camera=args.camera)

