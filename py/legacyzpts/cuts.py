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
# Sphinx build would crash
try:
    from astrometry.util.fits import fits_table, merge_tables
except ImportError:
    pass


CAMERAS= ['decam','90prime','mosaic']

def bit_codes():
# Bit codes for why a CCD got cut, used in cut_ccds().
    return dict(
        BLACKLIST = 0x1,
        BAD_EXPID = 0x2,
        CCDNAME_HDU_MISMATCH = 0x4,
        BAD_ASTROMETRY = 0x8,
        a = 0x10, # Mosaic3 one-third-pixel interpolation problem
        b = 0x20, # Mosaic3 one-third-pixel interpolation problem
        c = 0x40, # Mosaic3 one-third-pixel interpolation problem
        d = 0x80, # Mosaic3 one-third-pixel interpolation problem
    )


class LegacyzptsCuts(object):
    """Applies legacyzpts cuts

    Args:
        T_zpt: '*-zpts.fits' table
    """

    def __init__(self,T_zpt,camera=None):
        assert(camera in CAMERAS)
        self.camera= camera
        self.T= T_zpt.copy()
        self.good= np.ones(len(self.T),bool)
        self.df= pd.DataFrame({'err':big2small_endian(self.T.err_message),
                               'band':big2small_endian(self.T.filter)})
        self.df['err']= self.df['err'].str.strip()

    def err_message(self):
        self.good[self.df['err'].str.len() > 0]= False

    def third_pix(self):
        if self.camera == 'mosaic':
            # The 1/3-pixel shift problem was fixed in hardware on MJD 57674,
            # so only check for problems in data before then.
            self.good[((self.T.has_yshift == False) & 
                       (self.T.mjd_obs < 57674.))]= False

    def correct_band(self):
        self.good[self.df['band'].str.strip().isin(self.bands()) == False]= False

    def bands(self):
        return {'decam':['g','r','z'],
                'mosaic':['z'],
                '90prime':['g','r']
               }[self.camera]


class LegacypipeCuts(object):
    """Applies legacyipe cuts

    Args: 
        T_leg: '*-zpt.fits' table
    """
    def __init__(self,T_zpt,camera=None):
        assert(camera in CAMERAS)
        self.camera= camera
        self.ccds= create_legacypipe_table(T=T_zpt.copy(), camera=self.camera)
        self.good= np.ones(len(self.ccds),bool)
        # From legacypipe
        if self.camera == 'decam':
            self.ccds.ccdzpt += 2.5 * np.log10(self.ccds.exptime)

    def photometric(self):
        z0 = self.nominal_zeropoints()
        z0 = np.array([z0[f[0]] for f in self.ccds.filter])
        n0 = sum(self.good)
        # This is our list of cuts to remove non-photometric CCD images
        for name,crit in [
            ('exptime < 30 s', (self.ccds.exptime < 30)),
            ('ccdnmatch < 20', (self.ccds.ccdnmatch < 20)),
            ('abs(zpt - ccdzpt) > 0.1',
             (np.abs(self.ccds.zpt - self.ccds.ccdzpt) > 0.1)),
            ('zpt less than minimum',
             (self.ccds.zpt < self.min_zeropoint(z0))),
            ('zpt greater than maximum',
             (self.ccds.zpt > self.max_zeropoint(z0))),
        ]:
            self.good[crit] = False
            #continue as usual
            n = sum(self.good)
            print('Flagged', n0-n, 'more:',
                  name)
            n0 = n

    def bad_exposures(self,bad_expid_fn):
        if self.camera in ['90prime','mosaic']:
            bad_expids = np.loadtxt(bad_expid_fn, dtype=int, usecols=(0,))
            #import legacyccds
            #fn = os.path.join(os.path.dirname(legacyccds.__file__),
            #                  'bad_expid_mzls.txt')
            for expnum in bad_expids:
                self.good[self.ccds.expnum == expnum]= False

    def bright_sky(self):
        if self.camera == 'mosaic':
            for name,crit in [
                ('sky too bright', 
                 (self.ccds.ccdskycounts >= 150)),
            ]:
                self.good[crit] = False
                #continue as usual
                n = sum(self.ccd_cuts)
                print('Flagged', n0-n, 'more:',
                      name)
                n0 = n    


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
            
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--zpt_fn',action='store',required=True)
    args = parser.parse_args()

    T_zpt= fits_table(args.zpt_fn)
    isEmpty= T_zpt.filter == ' '
    if len(T_zpt[isEmpty]) > 0:
        print('Warning removing %d CCDs because filter is " "' % len(T_zpt[isEmpty]))
        T_zpt.cut(isEmpty == False)

    print('legacyzpts cuts')
    zpt_cuts= LegacyzptsCuts(T_zpt,args.camera)
    zpt_cuts.err_message()
    zpt_cuts.third_pix()
    zpt_cuts.correct_band()

    print('legacypipe cuts')
    leg_cuts= LegacypipeCuts(T_zpt,args.camera)
    leg_cuts.photometric()
    #leg_cuts.bad_exposures()
    print('Warning: skipping bad_exposures')
    leg_cuts.bright_sky()

    T_zpt.set('good',((zpt_cuts.good) & 
                      (leg_cuts.good)))
    fn=args.zpt_fn.replace('.fits.fz','').replace('.fits','') + 'survey-ccds-'+args.camera+'.fits'
    T_zpt.writeto(fn)
    print('Wrote %s' % fn)
    T_leg= create_legacypipe_table(T_zpt, camera=args.camera)
    T_leg.set('good',T_zpt.good)
    fn=fn.replace('.fits','-legacypipe.fits')
    T_zpt.writeto(fn)
    print('Wrote %s' % fn)
	
	

	
	
