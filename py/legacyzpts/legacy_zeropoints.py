from __future__ import division, print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pdb
import argparse

import numpy as np
from glob import glob
from pickle import dump
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
from scipy.ndimage.filters import median_filter
import re

import fitsio
from astropy.io import fits as fits_astropy
from astropy.table import Table, vstack
from astropy import units
from astropy.coordinates import SkyCoord
import datetime
import sys

from photutils import (CircularAperture, CircularAnnulus,
                       aperture_photometry, DAOStarFinder)

# Sphinx build would crash
try:
    from astrometry.util.starutil_numpy import hmsstring2ra, dmsstring2dec
    from astrometry.util.util import wcs_pv2sip_hdr
    from astrometry.util.ttime import Time
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.libkd.spherematch import match_radec
    from astrometry.libkd.spherematch import match_xy

    from tractor.splinesky import SplineSky

    from legacyanalysis.ps1cat import ps1cat
except ImportError:
    pass

CAMERAS=['decam','mosaic','90prime']
STAGING_CAMERAS={'decam':'decam',
                 'mosaic':'mosaicz',
                 '90prime':'bok'}


def try_mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass # Already exists, thats fine

def image_to_fits(img,fn,header=None,extname=None):
    fitsio.write(fn,img,header=header,extname=extname)
    print('Wrote %s' % fn)

def ptime(text,t0):
    tnow=Time()
    print('TIMING:%s ' % text,tnow-t0)
    return tnow

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    if len(lines) < 1: raise ValueError('lines not read properly from %s' % fn)
    return np.array( list(np.char.strip(lines)) )

def dobash(cmd):
    print('UNIX cmd: %s' % cmd)
    if os.system(cmd): raise ValueError


def extra_ccd_keys(camera='decam'):
    '''Returns list of camera-specific keywords for the ccd table'''
    if camera == 'decam':
        keys= [('ccdzpta', '>f4'), ('ccdzptb','>f4'), ('ccdnmatcha', '>i2'), ('ccdnmatchb', '>i2'),\
               ('temp', '>f4')]
    elif camera == 'mosaic':
        keys=[]
    elif camera == '90prime':
        keys=[('ccdzpt1', '>f4'), ('ccdzpt2','>f4'), ('ccdzpt3', '>f4'), ('ccdzpt4','>f4'),\
              ('ccdnmatcha', '>i2'), ('ccdnmatch2', '>i2'), ('ccdnmatch3', '>i2'), ('ccdnmatch4', '>i2')]
    return keys 

def get_units():
    return dict(
        ra='deg',dec='deg',exptime='sec',pixscale='arcsec/pix',
        fwhm='pix',seeing='arcsec',
        sky0='mag/arcsec^2',skymag='mag/arcsec^2/sec',
        skycounts='electron/pix/sec',skyrms='electron/pix/sec',
        apflux='electron/7 arcsec aperture',apskyflux='electron/7 arcsec aperture',
        apskyflux_perpix='electron/pix',
        apmags='-2.5log10(electron/sec) + zpt0',
        raoff='arcsec',decoff='arcsec',rarms='arcsec',decrms='arcsec',
        phoff='electron/sec',phrms='electron/sec',
        zpt0='electron/sec',zpt='electron/sec',transp='electron/sec')
 

def _ccds_table(camera='decam'):
    '''Initialize the CCDs table.

    Description and Units at:
    https://github.com/legacysurvey/legacyzpts/blob/master/DESCRIPTION_OF_OUTPUTS.md
    '''
    cols = [
        ('err_message', 'S30'), 
        ('image_filename', 'S100'), 
        ('image_hdu', '>i2'),      
        ('camera', 'S7'),          
        ('expnum', '>i4'),         
        ('ccdname', 'S4'),         
        ('ccdnum', '>i2'),        
        ('expid', 'S16'),        
        ('object', 'S35'),      
        ('propid', 'S10'),     
        ('filter', 'S1'),     
        ('exptime', '>f4'),  
        ('date_obs', 'S10'),
        ('mjd_obs', '>f8'),  
        ('ut', 'S15'),       
        ('ha', 'S13'),       
        ('airmass', '>f4'), 
        ('fwhm', '>f4'),       
        ('fwhm_cp', '>f4'),   
        ('gain', '>f4'),     
        ('width', '>i2'),   
        ('height', '>i2'), 
        ('ra_bore', '>f8'),     
        ('dec_bore', '>f8'),   
        ('crpix1', '>f4'),     
        ('crpix2', '>f4'),
        ('crval1', '>f8'),
        ('crval2', '>f8'),
        ('cd1_1', '>f4'),
        ('cd1_2', '>f4'),
        ('cd2_1', '>f4'),
        ('cd2_2', '>f4'),
        ('pixscale', 'f4'),   
        ('zptavg', '>f4'),   
        # -- CCD-level quantities --
        ('ra', '>f8'),        
        ('dec', '>f8'),      
        ('skymag', '>f4'),  
        ('skycounts', '>f4'),
        ('skyrms', '>f4'),
        ('sig1', '>f4'),
        ('nmatch_photom', '>i2'),   
        ('nmatch_astrom', '>i2'),  
        ('goodps1', '>i2'),   
        ('goodps1_wbadpix5', '>i2'),
        ('phoff', '>f4'),   
        ('phrms', '>f4'),  
        ('zpt', '>f4'),   
        ('zpt_wbadpix5', '>f4'), 
        ('transp', '>f4'),    
        ('raoff', '>f4'),    
        ('decoff', '>f4'),  
        ('rarms', '>f4'),  
        ('decrms', '>f4'),  
        ('rastddev', '>f4'),  
        ('decstddev', '>f4')  
        ]

    ccds = Table(np.zeros(1, dtype=cols))
    return ccds

     
def _stars_table(nstars=1):
    '''Initialize the stars table.

    Description and Units at:
    https://github.com/legacysurvey/legacyzpts/blob/master/DESCRIPTION_OF_OUTPUTS.md
    '''
    cols = [('image_filename', 'S100'),('image_hdu', '>i2'),
            ('expid', 'S16'), ('filter', 'S1'),('nmatch', '>i2'), 
            ('x', 'f4'), ('y', 'f4'),('expnum', '>i4'),
            ('gain', 'f4'),
            ('ra', 'f8'), ('dec', 'f8'), ('apmag', 'f4'),('apflux', 'f4'),('apskyflux', 'f4'),('apskyflux_perpix', 'f4'),
            ('radiff', 'f8'), ('decdiff', 'f8'),
            ('ps1_mag', 'f4'),
            ('gaia_g','f8'),('ps1_g','f8'),('ps1_r','f8'),('ps1_i','f8'),('ps1_z','f8'),
            ('exptime', '>f4')]
    stars = Table(np.zeros(nstars, dtype=cols))
    return stars

def reduce_survey_ccd_cols(survey_fn,legacy_fn):
    survey=fits_table(survey_fn)
    legacy=fits_table(legacy_fn)
    for col in survey.get_columns():
        if not col in legacy.get_columns():
            survey.delete_column(col)
    assert(len(legacy.get_columns()) == len(survey.get_columns()))
    for col in survey.get_columns():
        assert(col in legacy.get_columns())
    fn=survey_fn.replace('.fits.gz','_reduced.fits.gz')
    survey.writeto(fn) 
    print('Wrote %s' % fn)

def cuts_for_brick_2016p122(legacy_fn,survey_fn):
    survey=fits_table(survey_fn)
    legacy=fits_table(legacy_fn)
    # cut to same data as survey_fn
    keep= np.zeros(len(legacy),bool)
    for sur in survey:
        ind= (np.char.strip(legacy.image_filename) == sur.image_filename.strip() ) *\
             (np.char.strip(legacy.ccdname) == sur.ccdname.strip() )
        keep[ind]= True
    legacy.cut(keep)
    print('size legacy=%d' % (len(legacy),))
    # save
    fn=legacy_fn.replace('.fits','_wcuts.fits')
    legacy.writeto(fn) 
    print('Wrote %s' % fn)
     

def primary_hdr(fn):
    a= fitsio.FITS(fn)
    h= a[0].read_header()
    a.close()
    return h 

def get_pixscale(camera='decam'):
  assert(camera in CAMERAS)
  return {'decam':0.262,
          'mosaic':0.262,
          '90prime':0.470}[camera]

def run_create_legacypipe_table(zpt_list):
    fns= np.loadtxt(zpt_list,dtype=str)
    assert(len(fns) > 1)
    for fn in fns:
        create_legacypipe_table(fn)
        

def cols_for_legacypipe_table(which='all'):
    """Return list of -legacypipe.fits table colums

    Args:
        which: all, numeric, 
        nonzero_diff (numeric and expect non-zero diff with reference 
        when compute it)
    """
    assert(which in ['all','numeric','nonzero_diff'])
    if which == 'all':
        need_arjuns_keys= ['ra','dec','ra_bore','dec_bore',
                           'image_filename','image_hdu','expnum','ccdname','object',
                           'filter','exptime','camera','width','height','propid',
                           'mjd_obs','ccdnmatch',
                           'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff',
                           'ccdrarms', 'ccddecrms', 'ccdskycounts',
                           'ccdphrms',
                           'cd1_1','cd2_2','cd1_2','cd2_1',
                           'crval1','crval2','crpix1','crpix2']
        dustins_keys= ['skyrms', 'sig1']
    elif which == 'numeric':
        need_arjuns_keys= ['ra','dec','ra_bore','dec_bore',
                           'expnum',
                           'exptime','width','height',
                           'mjd_obs','ccdnmatch',
                           'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff',
                           'cd1_1','cd2_2','cd1_2','cd2_1',
                           'crval1','crval2','crpix1','crpix2']
        dustins_keys= ['skyrms']
    elif which == 'nonzero_diff':
        need_arjuns_keys= ['ra','dec','ccdnmatch',
                           'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff']
        dustins_keys= ['skyrms']
    return need_arjuns_keys + dustins_keys
 

def create_legacypipe_table(ccds_fn, camera=None):
    """input _ccds_table fn
    output a table formatted for legacypipe/runbrick
    """
    assert(camera in CAMERAS)
    need_keys= cols_for_legacypipe_table(which='all')
    # Load full zpt table
    assert('-zpt.fits' in ccds_fn)
    T = fits_table(ccds_fn)
    #hdr = T.get_header()
    #primhdr = fitsio.read_header(ccds_fn)
    #units= get_units()

    #primhdr.add_record(dict(name='ALLBANDS', value=allbands,
    #                        comment='Band order in array values'))
    #has_zpt = 'zpt' in T.columns()
    # Units
    if camera == 'decam':
        T.set('zpt',T.zpt - 2.5*np.log10(T.gain))
        T.set('zptavg',T.zptavg - 2.5*np.log10(T.gain))
    # Rename
    rename_keys= [('zpt','ccdzpt'),('zptavg','zpt'),
                  ('raoff','ccdraoff'),('decoff','ccddecoff'),
                  ('skycounts', 'ccdskycounts'),
                  ('rarms',  'ccdrarms'),
                  ('decrms', 'ccddecrms'),
                  ('phrms', 'ccdphrms'),
                  ('nmatch_photom','ccdnmatch')]
    for old,new in rename_keys:
        T.rename(old,new)
        #units[new]= units.pop(old)
    # Delete 
    del_keys= list( set(T.get_columns()).difference(set(need_keys)) )
    for key in del_keys:
        T.delete_column(key)
        #if key in units.keys():
        #    _= units.pop(key)
    # legacypipe/merge-zeropoints.py
    if camera == 'decam':
        T.set('width', np.zeros(len(T), np.int16) + 2046)
        T.set('height', np.zeros(len(T), np.int16) + 4094)
    # precision
    T.width  = T.width.astype(np.int16)
    T.height = T.height.astype(np.int16)
    #T.ccdnum = T.ccdnum.astype(np.int16) #number doesn't follow hdu, not using if possible
    T.cd1_1 = T.cd1_1.astype(np.float32)
    T.cd1_2 = T.cd1_2.astype(np.float32)
    T.cd2_1 = T.cd2_1.astype(np.float32)
    T.cd2_2 = T.cd2_2.astype(np.float32)
    # Align units with 'cols'
    #cols = T.get_columns()
    #units = [units.get(c, '') for c in cols]
    # Column ordering...
    #cols = []
    #if dr4:
    #    cols.append('release')
    #    T.release = np.zeros(len(T), np.int32) + 4000
    outfn=ccds_fn.replace('-zpt.fits','-legacypipe.fits')
    T.writeto(outfn) #, columns=cols, header=hdr, primheader=primhdr, units=units)
    print('Wrote %s' % outfn)


def cols_for_converted_star_table(star_table=None,
                                  which=None):
    assert(star_table in ['photom','astrom'])
    assert(which in ['all','numeric','nonzero_diff'])
    # which
    if which == 'all':
        need_arjuns_keys= ['filename','expnum','extname',
                           'ccd_x','ccd_y','ccd_ra','ccd_dec',
                           'ccd_mag','ccd_sky',
                           'raoff','decoff',
                           'magoff',
                           'nmatch',
                           'gmag','ps1_g','ps1_r','ps1_i','ps1_z']
        # If want it in star- table, add it here
        extra_keys= ['image_hdu','filter','ccdname'] 
    elif which == 'numeric':
        need_arjuns_keys= ['expnum',
                           'ccd_x','ccd_y','ccd_ra','ccd_dec',
                           'ccd_mag','ccd_sky',
                           'raoff','decoff',
                           'magoff',
                           'nmatch',
                           'gmag','ps1_g','ps1_r','ps1_i','ps1_z']
        extra_keys= [] 
    elif which == 'nonzero_diff':
        need_arjuns_keys= ['ccd_x','ccd_y','ccd_ra','ccd_dec',
                           'ccd_mag','ccd_sky',
                           'raoff','decoff',
                           'magoff',
                           'nmatch']
        extra_keys= [] 

    # star_table
    if star_table == 'photom':
        for key in ['raoff','decoff']:
            need_arjuns_keys.remove(key)

    elif star_table == 'astrom':
        for key in ['magoff']:
            need_arjuns_keys.remove(key)
    # Done 
    return need_arjuns_keys + extra_keys
 


def convert_stars_table(T, camera=None,
                        star_table=None):
    """converts -star.fits table to idl matches table

    Note, unlike converte_zeropoints_table, must treat each band 
        separately so loop over the bands

    Args:
        T: legacy stars fits_table, can be a single stars table or a merge
            of many stars tables
        camera: CAMERAS
        star_table: photom or astrom
    """
    assert(camera in CAMERAS)
    assert(star_table in ['photom','astrom'])
    from legacyzpts.qa.params import get_fiducial
    fid= get_fiducial(camera=camera)
    new_T= [] 
    for band in set(T.filter):
        isBand= T.filter == band
        zp0= fid.zp0[band]
        new_T.append(
            convert_stars_table_one_band(T[isBand],
                            camera= camera, star_table= star_table,
                            zp_fid=fid.zp0[band], 
                            pixscale=fid.pixscale))
    return merge_tables(new_T)

def convert_stars_table_one_band(T, camera=None, star_table=None,
                                 zp_fid=None,pixscale=0.262):
    """Converts legacy star fits table (T) to idl names and units
    
    Attributes:
        T: legacy star fits table
        star_table: photom or astrom
        zp_fid: fiducial zeropoint for the band
        pixscale: pixscale
        expnum2exptime: dict mapping expnum to exptime
    
    Example:
        kwargs= primary_hdr(zpt_fn)
        T= fits_table(stars_fn)
        newT= convert_stars_table(T, zp_fid=kwargs['zp_fid'],
        pixscale=kwargs['pixscale'])
    """ 
    assert(camera in CAMERAS)
    assert(star_table in ['photom','astrom'])
    assert(len(set(T.filter)) == 1)
    need_keys= cols_for_converted_star_table(
                        star_table=star_table,
                        which='all')
    extname=[ccdname for _,ccdname in np.char.split(T.expid,'-')]
    T.set('extname', np.array(extname))
    T.set('ccdname', np.array(extname))
    # AB mag of stars using fiducial ZP to convert
    #T.set('exptime', lookup_exptime(T.expnum, expnum2exptime))
    T.set('ccd_mag',-2.5 * np.log10(T.apflux / T.exptime) +  \
          zp_fid)
    # IDL matches- ccd_sky is counts / pix / sec where counts
    # is ADU for DECam and e- for mosaic/90prime 
    # legacyzpts star- apskyflux is e- from sky in 7'' aperture
    area= np.pi*3.5**2/pixscale**2
    if camera == 'decam':
        T.set('ccd_sky', T.apskyflux / area / T.gain)
    elif camera in ['mosaic','90prime']:
        T.set('ccd_sky', T.apskyflux / area / T.exptime)
    # Arjuns ccd_sky is ADUs in 7-10 arcsec sky aperture
    # e.g. sky (total e/pix/sec)= ccd_sky (ADU) * gain / exptime
    # Rename
    rename_keys= [('ra','ccd_ra'),('dec','ccd_dec'),('x','ccd_x'),('y','ccd_y'),
                  ('radiff','raoff'),('decdiff','decoff'),
                  ('dmagall','magoff'),
                  ('image_filename','filename'),
                  ('gaia_g','gmag')]
    if star_table == 'astrom':
        for key in [('dmagall','magoff')]:
            rename_keys.remove(key)
    for old,new in rename_keys:
        T.rename(old,new)
        #units[new]= units.pop(old)
    # Delete unneeded keys
    del_keys= list( set(T.get_columns()).difference(set(need_keys)) )
    for key in del_keys:
        T.delete_column(key)
        #if key in units.keys():
        #    _= units.pop(key)
    return T


def cols_for_converted_zpt_table(which='all'):
    """Return list of columns for -zpt.fits table converted to idl names

    Args:
        which: all, numeric, 
       nonzero_diff (numeric and expect non-zero diff with reference 
       when compute it)
    """
    assert(which in ['all','numeric','nonzero_diff'])
    if which == 'all':
        need_arjuns_keys= ['filename', 'object', 'expnum', 'exptime', 'filter', 'seeing', 'ra', 'dec', 
             'date_obs', 'mjd_obs', 'ut', 'ha', 'airmass', 'propid', 'zpt', 'avsky', 
             'fwhm', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 
             'naxis1', 'naxis2', 'ccdnum', 'ccdname', 'ccdra', 'ccddec', 
             'ccdzpt', 'ccdphoff', 'ccdphrms', 'ccdskyrms', 'ccdskymag', 
             'ccdskycounts', 'ccdraoff', 'ccddecoff', 'ccdrarms', 'ccddecrms', 'ccdtransp', 
             'ccdnmatch']

    elif which == 'numeric':
        need_arjuns_keys= ['expnum', 'exptime', 'seeing', 'ra', 'dec', 
             'mjd_obs', 'airmass', 'zpt', 'avsky', 
             'fwhm', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 
             'naxis1', 'naxis2', 'ccdnum', 'ccdra', 'ccddec', 
             'ccdzpt', 'ccdphoff', 'ccdphrms', 'ccdskyrms', 'ccdskymag', 
             'ccdskycounts', 'ccdraoff', 'ccddecoff', 'ccdrarms', 'ccddecrms', 'ccdtransp', 
             'ccdnmatch']

    elif which == 'nonzero_diff':
        need_arjuns_keys= ['seeing', 'ra', 'dec', 
             'zpt', 'avsky', 
             'fwhm', 
             'ccdra', 'ccddec', 
             'ccdzpt', 'ccdphoff', 'ccdphrms', 'ccdskyrms', 'ccdskymag', 
             'ccdskycounts', 'ccdraoff', 'ccddecoff', 'ccdrarms', 'ccddecrms', 'ccdtransp', 
             'ccdnmatch']

    return need_arjuns_keys
 

def convert_zeropoints_table(T, camera=None):
    """Make column names and units of -zpt.fits identical to IDL zeropoints

    Args:
        T: fits_table of some -zpt.fits like fits file
    """
    assert(camera in CAMERAS)
    pix= get_pixscale(camera)
    need_arjuns_keys= cols_for_converted_zpt_table(which='all')
    # Change units
    if camera == "decam":
        T.set('fwhm', T.fwhm * pix)
        T.set('skycounts', T.skycounts * T.exptime / T.gain)
        T.set('skyrms', T.skyrms * T.exptime / T.gain)
        T.set('zpt',T.zpt - 2.5*np.log10(T.gain))
        T.set('zptavg',T.zptavg - 2.5*np.log10(T.gain))
    elif camera in ['mosaic','90prime']:
        T.set('fwhm', T.fwhm * pix)
        T.set('fwhm_cp', T.fwhm_cp * pix)
    # Append 'ccd' to name
    app_ccd= ['skycounts','skyrms','skymag',
              'phoff','raoff','decoff',
              'phrms','rarms','decrms',
              'transp'] 
    for ad_ccd in app_ccd:
        T.rename(ad_ccd,'ccd'+ad_ccd)
    # Other
    rename_keys= [('ra','ccdra'),('dec','ccddec'),
                  ('ra_bore','ra'),('dec_bore','dec'),
                  ('fwhm','seeing'),('fwhm_cp','fwhm'),
                  ('zpt','ccdzpt'),('zptavg','zpt'),
                  ('width','naxis1'),('height','naxis2'),
                  ('image_filename','filename'),
                  ('nmatch_photom','ccdnmatch')]
    for old,new in rename_keys:
        T.rename(old,new)
    # New columns
    #T.set('avsky', np.zeros(len(T)) + np.mean(T.ccdskycounts))
    
    # Delete unneeded keys
    #needed= set(need_arjuns_keys).difference(set(ignoring_these))
    del_keys= list( set(T.get_columns()).difference(need_arjuns_keys) )
    for key in del_keys:
        T.delete_column(key)
    return T



#class NativeTable(object):
#    def __init__(self,fn,camera='decam',ccd_or_stars='ccds'):
#        '''zpt,stars tables have same units by default (e.g. electron/sec for zpt)
#        This func takes either the ccds or stars table and converts the relavent columns
#        into native units for given camera 
#        e.g. ADU for DECam,  electron/sec for Mosaic/BASS'''
#        assert(camera in ['decam','mosaic','90prime'])
#        assert(ccds_or_stars in ['ccds','stars'])
#        if camera in 'decam':
#            self.Decam(fn,ccds_or_stars=ccds_or_stars)
#        if camera in ['mosaic','90prime']:
#            self.Mosaic90Prime(fn,ccds_or_stars=ccds_or_stars)
#
#    def Decam(self,fn,ccds_or_stars):
#        T = fits_table(fn)
#        hdr = T.get_header()
#        primhdr = fitsio.read_header(ccds_fn)
#        units= get_units()
#        # Convert units
#        #T.set('zpt',T.zpt +- 2.5*np.log10(T.gain * T.exptime)) 
#        # Write
#        outfn=fn.replace('.fits','native.fits')
#        T.writeto(outfn, columns=cols, header=hdr, primheader=primhdr, units=units)

def getrms(x):
    return np.sqrt( np.mean( np.power(x,2) ) )

def get_bitmask_fn(imgfn):
    if 'ooi' in imgfn: 
        fn= imgfn.replace('ooi','ood')
    elif 'oki' in imgfn: 
        fn= imgfn.replace('oki','ood')
    else:
        raise ValueError('bad imgfn? no ooi or oki: %s' % imgfn)
    return fn

def get_weight_fn(imgfn):
    if 'ooi' in imgfn: 
        fn= imgfn.replace('ooi','oow')
    elif 'oki' in imgfn: 
        fn= imgfn.replace('oki','oow')
    else:
        raise ValueError('bad imgfn? no ooi or oki: %s' % imgfn)
    return fn

def get_90prime_expnum(primhdr):
    """converts 90prime header key DTACQNAM into the unique exposure number"""
    # /descache/bass/20160710/d7580.0144.fits --> 75800144
    base= (os.path.basename(primhdr['DTACQNAM'])
           .replace('.fits','')
           .replace('.fz',''))
    return int( re.sub(r'([a-z]+|\.+)','',base) )


class Measurer(object):
    """Main image processing functions for all cameras.

    Args:
        aprad: Aperture photometry radius in arcsec
        skyrad_inner,skyrad_outer: sky annulus in arcsec
        det_thresh: minimum S/N for matched filter
        match_radius: arcsec matching to gaia/ps1
        sn_min,sn_max: if not None then then {min,max} S/N will be enforced from 
            aperture photoemtry, where S/N = apflux/sqrt(skyflux)
        aper_sky_sub: do aperture sky subtraction instead of splinesky
    """

    def __init__(self, fn, aprad=3.5, skyrad_inner=7.0, skyrad_outer=10.0,
                 det_thresh=8., match_radius=3.,sn_min=None,sn_max=None,
                 aper_sky_sub=False, calibrate=False, **kwargs):
        # Set extra kwargs
        self.ps1_pattern= kwargs['ps1_pattern']
        self.ps1_gaia_pattern= kwargs['ps1_gaia_pattern']
        self.ps1_only= kwargs.get('ps1_only')
        
        self.zptsfile= kwargs.get('zptsfile')
        self.prefix= kwargs.get('prefix')
        self.verboseplots= kwargs.get('verboseplots')
        
        self.fn = fn
        self.debug= kwargs.get('debug')
        self.outdir= kwargs.get('outdir')

        if kwargs['psf']:
            if kwargs['calibdir']:
                self.calibdir= kwargs['calibdir']
            else:
                self.calibdir= os.getenv('LEGACY_SURVEY_DIR',None)
                if self.calibdir is None:
                    raise ValueError('LEGACY_SURVEY_DIR not set and --calibdir not given')
                self.calibdir= os.path.join(self.calibdir,'calib')

        self.aper_sky_sub = aper_sky_sub
        self.calibrate = calibrate
        
        self.aprad = aprad
        self.skyrad = (skyrad_inner, skyrad_outer)

        self.det_thresh = det_thresh    # [S/N] 
        self.match_radius = match_radius 
        self.sn_min = sn_min 
        self.sn_max = sn_max 
        
        # Tractor fitting of final star sample
        self.stampradius= 4. # [arcsec] Should be a bit bigger than radius=3.5'' aperture
        self.tractor_nstars= 30 # Tractorize at most this many stars, saves CPU time

        # Set the nominal detection FWHM (in pixels) and detection threshold.
        # Read the primary header and the header for this extension.
        self.nominal_fwhm = 5.0 # [pixels]
        
        try:
            self.primhdr = fitsio.read_header(fn, ext=0)
        except ValueError:
            # astropy can handle it
            tmp= fits_astropy.open(fn)
            self.primhdr= tmp[0].header
            tmp.close()
            del tmp
        # CP WCS succeed?
        assert('WCSCAL' in self.primhdr.keys())
        self.goodWcs=True  
        if not 'success' in self.primhdr['WCSCAL'].strip().lower():
            self.goodWcs=False  

        # Camera-agnostic primary header cards
        try:
            self.propid = self.primhdr['PROPID']
        except KeyError:
            self.propid = self.primhdr['DTPROPID']
        self.exptime = self.primhdr['EXPTIME']
        self.date_obs = self.primhdr['DATE-OBS']
        self.mjd_obs = self.primhdr['MJD-OBS']
        # Keys may not exist in header
        for key in ['AIRMASS','HA']:
            try:
                val= self.primhdr[key]
            except KeyError:
                val= -1
                print('WARNING! not in primhdr: %s' % key) 
            setattr(self, key.lower(),val)

        if kwargs['camera'] in ['decam','mosaic']:
            self.expnum= self.primhdr['EXPNUM']
        elif kwargs['camera'] == '90prime':
            self.expnum= get_90prime_expnum(self.primhdr)
        print('CP Header: EXPNUM = ',self.expnum)
        self.obj = self.primhdr['OBJECT']

    def zeropoint(self, band):
        return self.zp0[band]

    def sky(self, band):
        return self.sky0[band]

    def extinction(self, band):
        return self.k_ext[band]

    def set_hdu(self,ext):
        self.ext = ext.strip()
        self.ccdname= ext.strip()
        self.expid = '{:08d}-{}'.format(self.expnum, self.ccdname)
        hdulist= fitsio.FITS(self.fn)
        self.image_hdu= hdulist[ext].get_extnum() #NOT ccdnum in header!
        # use header
        self.hdr = fitsio.read_header(self.fn, ext=ext)
        # Sanity check
        assert(self.ccdname.upper() == self.hdr['EXTNAME'].strip().upper())
        self.ccdnum = np.int(self.hdr['CCDNUM']) 
        self.gain= self.get_gain(self.hdr)
        # WCS
        self.wcs = self.get_wcs()
        # Pixscale is assumed CONSTANT! per camera
        #self.pixscale = self.wcs.pixel_scale()

    def read_bitmask(self):
        dqfn= get_bitmask_fn(self.fn)
        mask = fitsio.read(dqfn, ext=self.ext)
        return mask

    def read_weight(self):
        fn= get_weight_fn(self.fn)
        wt = fitsio.read(fn, ext=self.ext)
        wt = self.scale_weight(wt)
        return wt

    def read_image(self):
        '''Read the image and header; scale the image.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        img = self.scale_image(img)
        return img, hdr

    def scale_image(self, img):
        return img

    def scale_weight(self, img):
        return img

    def remap_invvar(self, invvar, primhdr, img, dq):
        # By default, *do not* remap
        return invvar

    # A function that can be called by a subclasser's remap_invvar() method
    def remap_invvar_shotnoise(self, invvar, primhdr, img, dq):
        #
        # All three cameras scale the image and weight to units of electrons.
        #
        print('Remapping weight map for', self.fn)
        const_sky = primhdr['SKYADU'] # e/s, Recommended sky level keyword from Frank 
        expt = primhdr['EXPTIME'] # s
        with np.errstate(divide='ignore'):
            var_SR = 1./invvar # e**2

        print('median img:', np.median(img), 'vs sky estimate * exptime', const_sky*expt)

        var_Astro = np.abs(img - const_sky * expt) # img in electrons; Poisson process so variance = mean
        wt = 1./(var_SR + var_Astro) # 1/(e**2)
        # Zero out NaNs and masked pixels 
        wt[np.isfinite(wt) == False] = 0.
        wt[dq != 0] = 0.
        return wt

    def create_zero_one_mask(self,bitmask,good=[]):
        """Return zero_one_mask arraygiven a bad pixel map and good pix values
        bitmask: ood image
        good: list of values to treat as good in the bitmask
        """
        # 0 == good, 1 == bad
        zero_one_mask= bitmask.copy()
        for val in good:
            zero_one_mask[zero_one_mask == val]= 0 
        zero_one_mask[zero_one_mask > 0]= 1
        return zero_one_mask

    def get_zero_one_mask(self,bitmask,good=[]):
        """Convert bitmask into a zero and ones mask, 1 = bad, 0 = good
        bitmask: ood image
        good: (optional) list of values to treat as good in the bitmask
            default is to use appropiate values for the camera
        """
        # Defaults
        if len(good) == 0:
            if self.camera == 'decam':
                # 7 = transient
                good=[7]
            elif self.camera == 'mosaic':
                # 5 is truly a cosmic ray
                good=[]
            elif self.camera == '90prime':
                # 5 can be really bad for a good image because these are subtracted
                # and interpolated stats
                good= []
        return self.create_zero_one_mask(bitmask,good=good)

    def sensible_sigmaclip(self, arr, nsigma = 4.0):
        '''sigmaclip returns unclipped pixels, lo,hi, where lo,hi are the
        mean(goodpix) +- nsigma * sigma
        '''
        goodpix, lo, hi = sigmaclip(arr, low=nsigma, high=nsigma)
        meanval = np.mean(goodpix)
        sigma = (meanval - lo) / nsigma
        return meanval, sigma

    def get_sky_and_sigma(self, img, nsigma=3):
        '''returns 2d sky image and sky rms'''
        splinesky= False
        if splinesky:
            skyobj = SplineSky.BlantonMethod(img, None, 256)
            skyimg = np.zeros_like(img)
            skyobj.addTo(skyimg)
            mnsky, skystd = self.sensible_sigmaclip(img - skyimg,nsigma=nsigma)
            skymed= np.median(skyimg)
        else:
            #sky, sig1 = self.sensible_sigmaclip(img[1500:2500, 500:1000])
            if self.camera == 'decam':
                slc=[slice(1500,2500),slice(500,1500)]
            elif self.camera in ['mosaic','90prime']:
                slc=[slice(500,1500),slice(500,1500)]
            clip_vals,_,_ = sigmaclip(img[slc],low=nsigma,high=nsigma)
            # from astropy.stats import sigma_clip as sigmaclip_astropy
            #sky_masked= sigmaclip_astropy(img[slc],sigma=nsigma,iters=20)
            #use= sky1_masked.mask == False
            #skymed= np.median(sky_masked[use])
            #sky1std= np.std(sky_masked[use])
            skymed= np.median(clip_vals) 
            skystd= np.std(clip_vals) 
            skyimg= np.zeros(img.shape) + skymed
            # MAD gives 10% larger value
            # sig1= 1.4826 * np.median(np.abs(clip_vals))
        return skyimg, skymed, skystd

    def remove_sky_gradients(self, img):
        # Ugly removal of sky gradients by subtracting median in first x and then y
        H,W = img.shape
        meds = np.array([np.median(img[:,i]) for i in range(W)])
        meds = median_filter(meds, size=5)
        img -= meds[np.newaxis,:]
        meds = np.array([np.median(img[i,:]) for i in range(H)])
        meds = median_filter(meds, size=5)
        img -= meds[:,np.newaxis]

    def match_ps1_stars(self, px, py, fullx, fully, radius, stars):
        #print('Matching', len(px), 'PS1 and', len(fullx), 'detected stars with radius', radius)
        I,J,d = match_xy(px, py, fullx, fully, radius)
        #print(len(I), 'matches')
        dx = px[I] - fullx[J]
        dy = py[I] - fully[J]
        return I,J,dx,dy

    def fitstars(self, img, ierr, xstar, ystar, fluxstar):
        '''Fit each star using a Tractor model.'''
        import tractor

        H, W = img.shape

        fwhms = []
        radius_pix = self.stampradius / self.pixscale
                
        for ii, (xi, yi, fluxi) in enumerate(zip(xstar, ystar, fluxstar)):
            #print('Fitting source', i, 'of', len(Jf))
            ix = int(np.round(xi))
            iy = int(np.round(yi))
            xlo = int( max(0, ix-radius_pix) )
            xhi = int( min(W, ix+radius_pix+1) )
            ylo = int( max(0, iy-radius_pix) )
            yhi = int( min(H, iy+radius_pix+1) )
            xx, yy = np.meshgrid(np.arange(xlo, xhi), np.arange(ylo, yhi))
            r2 = (xx - xi)**2 + (yy - yi)**2
            keep = (r2 < radius_pix**2)
            pix = img[ylo:yhi, xlo:xhi].copy()
            ie = ierr[ylo:yhi, xlo:xhi].copy()
            #print('fitting source at', ix,iy)
            #print('number of active pixels:', np.sum(ie > 0), 'shape', ie.shape)

            psf = tractor.NCircularGaussianPSF([4.0], [1.0])
            tim = tractor.Image(data=pix, inverr=ie, psf=psf)
            src = tractor.PointSource(tractor.PixPos(xi-xlo, yi-ylo),
                                      tractor.Flux(fluxi))
            tr = tractor.Tractor([tim], [src])
        
            #print('Posterior before prior:', tr.getLogProb())
            src.pos.addGaussianPrior('x', 0.0, 1.0)
            #print('Posterior after prior:', tr.getLogProb())
                
            tim.freezeAllBut('psf')
            psf.freezeAllBut('sigmas')
        
            # print('Optimizing params:')
            # tr.printThawedParams()
        
            #print('Parameter step sizes:', tr.getStepSizes())
            optargs = dict(priors=False, shared_params=False)
            for step in range(50):
                dlnp, x, alpha = tr.optimize(**optargs)
                #print('dlnp', dlnp)
                #print('src', src)
                #print('psf', psf)
                if dlnp == 0:
                    break
                
            # Now fit only the PSF size
            tr.freezeParam('catalog')
            # print('Optimizing params:')
            # tr.printThawedParams()
        
            for step in range(50):
                dlnp, x, alpha = tr.optimize(**optargs)
                #print('dlnp', dlnp)
                #print('src', src)
                #print('psf', psf)
                if dlnp == 0:
                    break

            fwhms.append(2.35 * psf.sigmas[0]) # [pixels]
            #model = tr.getModelImage(0)
            #pdb.set_trace()
        
        return np.array(fwhms)

    def isolated_radec(self,ra,dec,nn=2,minsep=1./3600):
        '''return indices of ra,dec for which the ra,dec points are 
        AT LEAST a distance minsep away from their nearest neighbor point'''
        cat1 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
        cat2 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
        idx, d2d, d3d = cat1.match_to_catalog_3d(cat2,nthneighbor=nn)
        b= np.array(d2d) >= minsep
        return b

    def get_ps1_cuts(self,ps1):
        """Returns bool of PS1 sources to keep
        ps1: catalogue with ps1 data
        """
        gicolor= ps1.median[:,0] - ps1.median[:,2]
        return ((ps1.nmag_ok[:, 0] > 0) &
                (ps1.nmag_ok[:, 1] > 0) &
                (ps1.nmag_ok[:, 2] > 0) &
                (gicolor > 0.4) & 
                (gicolor < 2.7))
   
    def return_on_error(self,err_message='',
                        ccds=None, stars_photom=None, stars_astrom=None):
        """Sets ccds table err message, zpt to nan, and returns appropriately for self.run() 
        
        Args: 
         err_message: length <= 30 
         ccds, stars_photom, stars_astrom: (optional) tables partially filled by run() 
        """
        assert(len(err_message) > 0 & len(err_message) <= 30)
        if ccds is None:
            ccds= _ccds_table(self.camera)
        if stars_photom is None:
            stars_photom= _stars_table()
        if stars_astrom is None:
            stars_astrom= _stars_table()
        ccds['err_message']= err_message
        ccds['zpt']= np.nan
        return ccds, stars_photom, stars_astrom

    def run(self, ext=None, save_xy=False, psfex=False, splinesky=False):
        """Computes statistics for 1 CCD
        
        Args: 
            ext: ccdname
            save_xy: save daophot x,y and x,y after various cuts to dict and save
                to json
        
        Returns:
            ccds, stars_photom, stars_astrom
        """
        if not self.goodWcs:
            print('WCS Failed')
            return self.return_on_error(err_message='WCS Failed')
        self.set_hdu(ext)
        # 
        t0= Time()
        t0= ptime('Measuring CCD=%s from image=%s' % (self.ccdname,self.fn),t0)

        # Initialize 
        ccds = _ccds_table(self.camera)
        if STAGING_CAMERAS[self.camera] in self.fn:
            ccds['image_filename'] = self.fn[self.fn.rfind('/%s/' % \
                                           STAGING_CAMERAS[self.camera])+1:]
        else:
            # img not on proj
            ccds['image_filename'] = self.fn #os.path.basename(self.fn)
        ccds['image_hdu'] = self.image_hdu 
        ccds['ccdnum'] = self.ccdnum 
        ccds['camera'] = self.camera
        ccds['expnum'] = self.expnum
        ccds['ccdname'] = self.ccdname
        ccds['expid'] = self.expid
        ccds['object'] = self.obj
        ccds['propid'] = self.propid
        ccds['filter'] = self.band
        ccds['exptime'] = self.exptime
        ccds['date_obs'] = self.date_obs
        ccds['mjd_obs'] = self.mjd_obs
        ccds['ut'] = self.ut
        ccds['ra_bore'] = self.ra_bore
        ccds['dec_bore'] = self.dec_bore
        ccds['ha'] = self.ha
        ccds['airmass'] = self.airmass
        ccds['gain'] = self.gain
        ccds['pixscale'] = self.pixscale
        
        # From CP Header
        hdrVal={}
        # values we want
        for ccd_col in ['width','height','fwhm_cp']:
            # Possible keys in hdr for these values
            for key in self.cp_header_keys[ccd_col]:
                if key in self.hdr.keys():
                    hdrVal[ccd_col]= self.hdr[key]
                    break
        for ccd_col in ['width','height','fwhm_cp']:
            if ccd_col in hdrVal.keys():
                print('CP Header: %s = ' % ccd_col,hdrVal[ccd_col])
                ccds[ccd_col]= hdrVal[ccd_col]
            else:
                warning='Could not find %s, keys not in cp header: %s' % \
                        (ccd_col,self.cp_header_keys[ccd_col])
                if ccd_col == 'fwhm_cp':
                    print('WARNING: %s' % warning)
                    ccds[ccd_col]= np.nan
                else:
                    raise KeyError(warning)
        hdr_fwhm= ccds['fwhm_cp'].data[0]
        #hdr_fwhm=-1
        #for fwhm_key in self.cp_fwhm_keys:
        #  if fwhm_key in hdr.keys():
        #    hdr_fwhm= hdr[fwhm_key] #FWHM in pixels
        #    break
        #if hdr_fwhm < 0:
        #    ccds['fwhm_cp']= hdr_fwhm # -1 so know didn't find it
        #    hdr_fwhm= 1.3 / self.pixscale #fallback value for source detection
        #else:
        #    ccds['fwhm_cp']= hdr_fwhm
        # Same ccd and header names
        notneeded_cols= ['avsky']
        for ccd_col in ['avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 
                        'cd1_1','cd1_2', 'cd2_1', 'cd2_2']:
            if ccd_col.upper() in self.hdr.keys():
                print('CP Header: %s = ' % ccd_col,self.hdr[ccd_col])
                ccds[ccd_col]= self.hdr[ccd_col]
            else:
                if ccd_col in notneeded_cols:
                    ccds[ccd_col]= np.nan
                else:
                    raise KeyError('Could not find %s, keys not in cp header:' \
                               % ccd_col,ccd_col)
        #hdrkey = ('avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
        #          'cd1_2', 'cd2_1', 'cd2_2')
        #ccdskey = ('avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
        #           'cd1_2', 'cd2_1', 'cd2_2')
        #for ckey, hkey in zip(ccdskey, hdrkey):
        #    try:
        #        ccds[ckey] = hdr[hkey]
        #    except KeyError:
        #        #if hkey == 'avsky':
        #        #    print('CP image does not have avsky in hdr: %s' % ccds['image_filename'])
        #        #    ccds[hkey]= -1
        #        raise NameError('key not in header: %s' % hkey)
            
        exptime = ccds['exptime'].data[0]
        airmass = ccds['airmass'].data[0]
        print('Band {}, Exptime {}, Airmass {}'.format(self.band, exptime, airmass))

        # WCS: 1-indexed so pixel pixelxy2radec(1,1) corresponds to img[0,0]
        H = ccds['height'].data[0]
        W = ccds['width'].data[0]
        print('Image size:', W,H)
        ccdra, ccddec = self.wcs.pixelxy2radec((W+1) / 2.0, (H + 1) / 2.0)
        ccds['ra'] = ccdra   # [degree]
        ccds['dec'] = ccddec # [degree]
        t0= ptime('header-info',t0)

        # Test WCS again IDL, WCS is 1-indexed
        #x_pix= [1,img.shape[0]/2,img.shape[0]]
        #y_pix= [1,img.shape[1]/2,img.shape[1]]
        #test_wcs= [(_x,_y)+self.wcs.pixelxy2radec(_x,_y) for _x,_y in zip(x_pix,y_pix)]
        #with open('three_camera_vals.txt','a') as foo:
        #    foo.write('ccdname=%s, hdu=%d, image=%s\n' % (self.ccdname,self.image_hdu,self.fn))
        #    foo.write('image shape: x=%d y=%d\n' % (img.shape[0],img.shape[1]))
        #    for i in test_wcs:
        #        foo.write('x=%d y=%d ra=%.9f dec=%.9f\n' % (i[0],i[1],i[2],i[3]))
        #return ccds, _stars_table()
        
        if (self.camera == 'decam') & (ext == 'S7'):
            return self.return_on_error(err_message='S7', ccds=ccds)

        self.img,hdr= self.read_image() 
        self.bitmask= self.read_bitmask()
        weight = self.read_weight()

        # Per-pixel error -- weight is 1/sig*2, scaled by scale_weight()
        medweight = np.median(weight[(weight > 0) * (self.bitmask == 0)])
        # Undo the weight scaling to get sig1 back into native image units
        wscale = self.scale_weight(1.)
        ccds['sig1'] = 1. / np.sqrt(medweight / wscale)

        self.invvar = self.remap_invvar(weight, self.primhdr, self.img, self.bitmask)

        t0= ptime('read image',t0)

        # Measure the sky brightness and (sky) noise level.  Need to capture
        # negative sky.
        sky0 = self.sky(self.band)
        zp0 = self.zeropoint(self.band)
        print('Computing the sky background.')
        sky_img, skymed, skyrms = self.get_sky_and_sigma(self.img)
        img_sub_sky= self.img - sky_img

        #fn= 'N4.fits' 
        #fitsio.write(fn,img_sub_sky,extname='N4')
        #raise ValueError
        

        # Bunch of sky estimates
        # Median of absolute deviation (MAD), std dev = 1.4826 * MAD
        print('sky from median of image= %.2f' % skymed)
        skybr = zp0 - 2.5*np.log10(skymed / self.pixscale / self.pixscale / exptime)
        print('  Sky brightness: {:.3f} mag/arcsec^2'.format(skybr))
        print('  Fiducial:       {:.3f} mag/arcsec^2'.format(sky0))

        ccds['skyrms'] = skyrms / exptime # e/sec
        ccds['skycounts'] = skymed / exptime # [electron/pix]
        ccds['skymag'] = skybr   # [mag/arcsec^2]
        t0= ptime('measure-sky',t0)

        # Load PS1 and PS1-Gaia Catalogues 
        # We will only used detected sources that have PS1 or PS1-gaia matches
        # So cut to this super set immediately
        
        #pattern={'ps1':'/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-%(hp)05d.fits',
        #         'ps1_gaia':'/project/projectdirs/cosmo/work/gaia/chunks-ps1-gaia/chunk-%(hp)05d.fits'}
        #ps1_pattern= #os.environ["PS1CAT_DIR"]=PS1
        #ps1_gaia_patternos.environ["PS1_GAIA_MATCHES"]= PS1_GAIA_MATCHES
        try:
            ps1 = ps1cat(ccdwcs=self.wcs, 
                         pattern= self.ps1_pattern).get_stars(magrange=None)
            ps1_gaia = ps1cat(ccdwcs=self.wcs,
                              pattern= self.ps1_gaia_pattern).get_stars(magrange=None)
        except OSError:
            txt="outside PS1 footprint,In Gal. Plane"
            print(txt)
            return self.return_on_error(mess,ccds=ccds)
        assert(len(ps1_gaia.columns()) > len(ps1.columns())) 
        ps1band = ps1cat.ps1band[self.band]
        # PS1 cuts
        if len(ps1):
            ps1.cut( self.get_ps1_cuts(ps1) )
            # Convert to Legacy Survey mags
            colorterm = self.colorterm_ps1_to_observed(ps1.median, self.band)
            ps1.legacy_survey_mag = ps1.median[:, ps1band] + colorterm
        if len(ps1_gaia):
            ps1_gaia.cut( self.get_ps1_cuts(ps1_gaia) )
            # Add gaia ra,dec
            ps1_gaia.gaia_dec = ps1_gaia.dec_ok - ps1_gaia.ddec/3600000.
            ps1_gaia.gaia_ra  = (ps1_gaia.ra_ok - 
                         ps1_gaia.dra/3600000./np.cos(np.deg2rad(ps1_gaia.gaia_dec)))
            # same for ps1_gaia -- but clip the color term because we don't clip the g-i color.
            colorterm = self.colorterm_ps1_to_observed(ps1_gaia.median, self.band)
            ps1_gaia.legacy_survey_mag = ps1_gaia.median[:, ps1band] + np.clip(colorterm, -1., +1.)
        
        if not psfex:
            # badpix5 test, all good PS1 
            if self.camera in ['90prime','mosaic']:
                _, ps1_x, ps1_y = self.wcs.radec2pixelxy(ps1.ra_ok,ps1.dec_ok)
                ps1_x-= 1.
                ps1_y-= 1.
                ap_for_ps1 = CircularAperture((ps1_x, ps1_y), 5.)
                # special mask, only gt 0 where badpix eq 5
                img_mask_5= np.zeros(self.bitmask.shape, dtype=self.bitmask.dtype)
                img_mask_5[self.bitmask == 5]= 1
                phot_for_mask_5 = aperture_photometry(img_mask_5, ap_for_ps1)
                flux_for_mask_5 = phot_for_mask_5['aperture_sum'] 
                ccds['goodps1']= len(ps1)
                ccds['goodps1_wbadpix5']= len(ps1[flux_for_mask_5.data > 0])

            # Detect stars on the image.  
            # 10 sigma, sharpness, roundness all same as IDL zeropoints (also the defaults)
            # Exclude_border=True removes the stars with centroid on or out of ccd edge
            # Good, but we want to remove with aperture touching ccd edge too
            print('det_thresh = %d' % self.det_thresh)
            #threshold=self.det_thresh * stddev_mad,
            dao = DAOStarFinder(fwhm= hdr_fwhm,
                                threshold=self.det_thresh * skyrms,
                                sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                                exclude_border=False)
            obj= dao(self.img)
            if len(obj) < self.minstar:
                dao.threshold /= 2.
                obj= dao(self.img)
                if len(obj) < self.minstar:
                    return self.return_on_error('dao found < %d sources' % self.minstar,ccds=ccds)
            t0= ptime('detect-stars',t0)
    
            # We for sure know that sources near edge could be bad
            edge_sep = 1. + self.skyrad[1] 
            edge_sep_px = edge_sep/self.pixscale
            wid,ht= self.img.shape[1],self.img.shape[0] 
            away_from_edge= (
                     (obj['xcentroid'] > edge_sep_px) &
                     (obj['xcentroid'] < wid - edge_sep_px) &
                     (obj['ycentroid'] > edge_sep_px) &
                     (obj['ycentroid'] < ht - edge_sep_px))
            obj= obj[away_from_edge] 
            objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
            nobj = len(obj)
            print('{} sources detected with detection threshold {}-sigma minus edge sources'.format(nobj, self.det_thresh))
            ccds['nstarfind']= nobj
            if nobj < self.minstar:
                return self.return_on_error('after edge cuts < %d sources' % self.minstar,ccds=ccds)
    
            if save_xy:
              # Arrays of length number of all daophot found sources
              all_xy= fits_table()
              all_xy.set('x', obj['xcentroid'].data)
              all_xy.set('y', obj['ycentroid'].data)
              all_xy.set('ra', objra)
              all_xy.set('dec', objdec)
              all_xy.writeto('%s_%s_all_xy.fits' % 
                      (os.path.basename(self.fn).replace('.fits','').replace('.fz',''),
                       ext))
         

            # Matching
            matched= {}
            # Photometry
            matched['photom_obj'], matched['photom_ref'], _ = \
                          match_radec(objra, objdec, ps1.ra_ok, ps1.dec_ok, 
                                      self.match_radius/3600.0,
                                      nearest=True)
            t0= ptime('matching-for-photometer',t0)
            if len(matched['photom_obj']) < self.minstar:
                return self.return_on_error('photom matched < %d sources' % self.minstar,ccds=ccds)
            stars_photom,err= self.do_Photometry(obj[matched['photom_obj']],
                                             ps1[matched['photom_ref']],
                                             ccds=ccds, save_xy=save_xy)
            if len(err) > 0:
                return self.return_on_error(err,ccds=ccds,
                                            stars_photom=stars_photom)
            t0= ptime('photutils-photometry',t0)
            
            # Astrometry
            matched['astrom_obj'], matched['astrom_ref'], _ = \
                             match_radec(objra, objdec, ps1_gaia.gaia_ra, ps1_gaia.gaia_dec, 
                                         self.match_radius/3600.0,
                                         nearest=True)
            t0= ptime('matching-for-astrometry',t0)
            if ((self.ps1_only) |
                (len(matched['astrom_obj']) < 20)):
              # Either have Gaia pot holes or are forcing PS1
              # use ps1
              stars_astrom,err= self.do_Astrometry(
                                   obj[matched['photom_obj']],
                                   ref_ra= ps1.ra_ok[matched['photom_ref']],
                                   ref_dec= ps1.dec_ok[matched['photom_ref']],
                                   ccds=ccds)
            else:
              # Use gaia
              if len(matched['astrom_obj']) < self.minstar:
                  return self.return_on_error('astrom gaia matched < %d sources' % self.minstar,ccds=ccds,stars_photom=stars_photom)
              stars_astrom,err= self.do_Astrometry(
                                   obj[matched['astrom_obj']],
                                   ref_ra= ps1_gaia.gaia_ra[matched['astrom_ref']],
                                   ref_dec= ps1_gaia.gaia_dec[matched['astrom_ref']],
                                   ccds=ccds)
            if len(err) > 0:
                return self.return_on_error(err,ccds=ccds,
                                            stars_photom=stars_photom,
                                            stars_astrom=stars_astrom)
            t0= ptime('did-astrometry',t0)
            
            # FWHM
            # Tractor on specific SN sources 
            ap = CircularAperture((stars_photom['x'], stars_photom['y']), 
                                   self.aprad / self.pixscale)
            skyphot = aperture_photometry(sky_img, ap)
            skyflux = skyphot['aperture_sum'].data
            star_SN= stars_photom['apflux'].data / np.sqrt(stars_photom['apflux'].data + skyflux)
            t0= ptime('photutils-photometry-SN',t0)
     
            # Brightest N stars
            sn_cut= ((star_SN >= 10.) &
                     (star_SN <= 100.))
            if len(star_SN[sn_cut]) < 10.:
                sn_cut= star_SN >= 10.
                if len(star_SN[sn_cut]) < 10.:
                    sn_cut= np.ones(len(star_SN),bool)
            i_low_hi= np.argsort(star_SN)[sn_cut] 
            # brightest stars in sample, at most self.tractor_nstars
            sample=dict(x= stars_photom['x'][i_low_hi][-self.tractor_nstars:],
                        y= stars_photom['y'][i_low_hi][-self.tractor_nstars:],
                        apflux= stars_photom['apflux'][i_low_hi][-self.tractor_nstars:],
                        sn= star_SN[i_low_hi][-self.tractor_nstars:])
            #ivar = np.zeros_like(img) + 1.0/sig1**2
            # Hack! To avoid 1/0 and sqrt(<0) just considering Poisson Stats due to sky
            ierr = 1.0/np.sqrt(sky_img)
            fwhms = self.fitstars(img_sub_sky, ierr, sample['x'], sample['y'], sample['apflux'])
            ccds['fwhm'] = np.median(fwhms) # fwhms= 2.35 * psf.sigmas 
            print('FWHM med=%f, std=%f, std_med=%f' % (np.median(fwhms),np.std(fwhms),np.std(fwhms)/len(sample['x'])))
            #ccds['seeing'] = self.pixscale * np.median(fwhms)
            t0= ptime('Tractor fit FWHM to %d/%d stars' % (len(sample['x']),len(stars_photom)), t0) 
              
            # RESULTS
            print("RESULTS %s" % ext)
            print('Photometry: %d stars' % ccds['nmatch_photom'])
            print('Offset (mag) =%.4f, rms=0.4%f' % (ccds['phoff'],ccds['phrms'])) 
            print('Zeropoint %.4f' % (ccds['zpt'],))
            print('Transparency %.4f' % (ccds['transp'],))
            print('Astrometry: %d stars' % ccds['nmatch_astrom'])
            print('Offsets (arcsec) RA=%.6f, Dec=%.6f' % (ccds['raoff'], ccds['decoff'])) 
    
        else: # psfex
            # Now put Gaia stars into the image and re-fit their centroids
            # and fluxes using the tractor with the PsfEx PSF model.
    
            # assume that the CP WCS has gotten us to within a few pixels
            # of the right answer.  Find Gaia stars, initialize Tractor
            # sources there, optimize them and see how much they want to
            # move.
            psf = self.get_psfex_model()
            ccds['fwhm'] = psf.fwhm

            fit_img = img_sub_sky

            if splinesky:
                sky = self.get_splinesky()
                print('Instantiating and subtracting sky model')
                skymod = np.zeros_like(self.img)
                sky.addTo(skymod)
                # Apply the same transformation that was applied to the image...
                skymod = self.scale_image(skymod)

                print('Old sky_img: avg', np.mean(sky_img), 'min/max', np.min(sky_img), np.max(sky_img))
                print('Skymod: avg', np.mean(skymod), 'min/max', skymod.min(), skymod.max())

                fit_img = self.img - skymod


            # PS1 for photometry

            # Initial flux estimate, from nominal zeropoint
            flux0 = 10.**((zp0 - ps1.legacy_survey_mag) / 2.5) * exptime

            ierr = np.sqrt(self.invvar)

            # plt.clf()
            # n,b,p = plt.hist((fit_img * ierr)[ierr > 0], range=(-6,6), bins=100)
            # plt.xlabel('Image pixel chi')
            # xx = np.linspace(-6,6, 100)
            # yy = 1./np.sqrt(2.*np.pi) * np.exp(-0.5 * xx**2)
            # yy *= sum(n)
            # db = b[1]-b[0]
            # plt.plot(xx, yy * db, 'r-')
            # plt.xlim(-6,6)
            # fn = 'chi-%i-%s.png' % (self.expnum, self.ccdname)
            # plt.savefig(fn)
            # print('Wrote', fn)
            # 
            # plt.clf()
            # I = (ierr > 0) * (fit_img > 1)
            # plt.hexbin(fit_img[I], 1. / ierr[I],
            #            xscale='log', yscale='log')
            # plt.xlabel('Image pixel value')
            # plt.ylabel('Uncertainty')
            # fn = 'unc-%i-%s.png' % (self.expnum, self.ccdname)
            # plt.savefig(fn)
            # print('Wrote', fn)

            # Run tractor fitting of the PS1 stars, using the PsfEx model.
            phot = self.tractor_fit_sources(ps1.ra_ok, ps1.dec_ok, flux0,
                                            fit_img, ierr, psf)
            ref = ps1[phot.iref]
            phot.delete_column('iref')
            ref.rename('ra_ok',  'ra')
            ref.rename('dec_ok', 'dec')

            phot.raoff  = (ref.ra  - phot.ra_fit ) * 3600. * np.cos(np.deg2rad(ref.dec))
            phot.decoff = (ref.dec - phot.dec_fit) * 3600.

            ok, = np.nonzero(phot.flux > 0)
            phot.psfmag = np.zeros(len(phot), np.float32)
            phot.psfmag[ok] = -2.5*np.log10(phot.flux[ok] / exptime) + zp0
            phot.dpsfmag = np.zeros_like(phot.psfmag)
            phot.dpsfmag[ok] = np.abs((-2.5 / np.log(10.)) * phot.dflux[ok] / phot.flux[ok])

            H,W = self.bitmask.shape
            phot.bitmask = self.bitmask[np.clip(phot.y1, 0, H-1).astype(int),
                                        np.clip(phot.x1, 0, W-1).astype(int)]
    
            dmagall = ref.legacy_survey_mag - phot.psfmag
            if not np.all(np.isfinite(dmagall)):
                print(np.sum(np.logical_not(np.isfinite(dmagall))), 'stars have NaN mags; ignoring')
                dmagall = dmagall[np.isfinite(dmagall)]
                print('Continuing with', len(dmagall), 'stars')

            dmag, _, _ = sigmaclip(dmagall, low=2.5, high=2.5)
            ndmag = len(dmag)
            dmagmed = np.median(dmag)
            dmagsig = np.std(dmag)
            zptmed = zp0 + dmagmed
            kext = self.extinction(self.band)
            transp = 10.**(-0.4 * (zp0 - zptmed - kext * (airmass - 1.0)))
    
            raoff = np.median(phot.raoff)
            decoff = np.median(phot.decoff)

            print('Tractor PsfEx-fitting results for PS1 (photometry):')
            print('RA, Dec offsets (arcsec) relative to PS1: %.4f, %.4f' %
                  (raoff, decoff))
            print('RA, Dec stddev (arcsec): %.4f, %.4f' %
                  (np.std(phot.raoff), np.std(phot.decoff)))
            print('Mag offset: %.4f' % dmagmed)
            print('Scatter: %.4f' % dmagsig)
            print('Number stars used for zeropoint median %d' % ndmag)
            print('Zeropoint %.4f' % zptmed)
            print('Transparency %.4f' % transp)

            for c in ['x0','y0','x1','y1','flux','raoff','decoff', 'psfmag',
                      'dflux','dx','dy']:
                phot.set(c, phot.get(c).astype(np.float32))

            phot.ra_ps1  = ref.ra
            phot.dec_ps1 = ref.dec
            phot.ps1_mag = ref.legacy_survey_mag
            for band in 'griz':
                i = ps1cat.ps1band.get(band, None)
                if i is None:
                    continue
                phot.set('ps1_'+band, ref.median[:,i].astype(np.float32))
            # Save CCD-level information in the per-star table.
            phot.ccd_raoff  = np.zeros(len(phot), np.float32) + raoff
            phot.ccd_decoff = np.zeros(len(phot), np.float32) + decoff
            phot.ccd_phoff  = np.zeros(len(phot), np.float32) + dmagmed
            phot.ccd_zpt    = np.zeros(len(phot), np.float32) + zptmed
            phot.expnum = np.zeros(len(phot), np.int32) + self.expnum
            phot.ccdname = np.array([self.ccdname] * len(phot))
            phot.exptime = np.zeros(len(phot), np.float32) + self.exptime
            phot.gain = np.zeros(len(phot), np.float32) + self.gain

            # Convert to astropy Table
            cols = phot.get_columns()
            stars_photom = Table([phot.get(c) for c in cols], names=cols)

            # Add to the zeropoints table
            ccds['raoff']  = raoff
            ccds['decoff'] = decoff
            ccds['rastddev'] = np.std(phot.raoff)
            ccds['decstddev'] = np.std(phot.decoff)
            ra_clip, _, _ = sigmaclip(phot.raoff, low=3., high=3.)
            ccds['rarms'] = getrms(ra_clip)
            dec_clip, _, _ = sigmaclip(phot.decoff, low=3., high=3.)
            ccds['decrms'] = getrms(dec_clip)
            ccds['phoff'] = dmagmed
            ccds['phrms'] = dmagsig
            ccds['zpt'] = zptmed
            ccds['transp'] = transp       
            ccds['nmatch_photom'] = len(phot)
            ccds['nmatch_astrom'] = len(phot)

            # Astrometry
            if self.ps1_only or len(ps1_gaia) < 20:
                # Keep the PS1 results for astrometry
                if not self.ps1_only:
                    print('PS1/Gaia match catalog has only', len(ps1_gaia), 'stars - using PS1 for astrometry')
                stars_astrom = None
            else:
                # Fast-track "phot" results within 1".
                # That is, match the "ps1_gaia" sample with the sample that we just fit during the photometry phase,
                # and use those results for astrometry, rather than re-fitting the same stars.
                I,J,d = match_radec(ps1_gaia.gaia_ra, ps1_gaia.gaia_dec,
                                    phot.ra_fit, phot.dec_fit, 1./3600.,
                                    nearest=True)
                print(len(I), 'of', len(ps1_gaia), 'PS1/Gaia sources have a match in PS1')
                
                fits = []
                refs = []
                if len(I):
                    photmatch = fits_table()
                    for col in ['x0','y0','x1','y1','flux','psfsum','ra_fit','dec_fit',
                                'dflux','dx','dy']:
                        photmatch.set(col, phot.get(col)[J])
                    # Use as reference catalog the PS1-Gaia sources that matched the PS1 stars.
                    photref = ps1_gaia[I]
                    # Update the "x0","y0" columns to be the X,Y
                    # coords from pushing the *gaia* RA,Decs through
                    # the WCS.
                    ok,xx,yy = self.wcs.radec2pixelxy(photref.gaia_ra, photref.gaia_dec)
                    photmatch.x0 = xx - 1.
                    photmatch.y0 = yy - 1.
                    fits.append(photmatch)
                    refs.append(photref)

                    # Now take the PS1-Gaia stars that didn't have a match and fit them.
                    unmatched = np.ones(len(ps1_gaia), bool)
                    unmatched[I] = False
                    ps1_gaia.cut(unmatched)

                if len(ps1_gaia):
                    # If there were PS1-Gaia stars that didn't match to PS1, fit them now...
                    flux0 = 10.**((zp0 - ps1_gaia.legacy_survey_mag) / 2.5) * exptime
                    astrom = self.tractor_fit_sources(ps1_gaia.gaia_ra, ps1_gaia.gaia_dec, flux0,
                                                      fit_img, ierr, psf)
                    if len(astrom):
                        ref = ps1_gaia[astrom.iref]
                        astrom.delete_column('iref')
                        fits.append(astrom)
                        refs.append(ref)

                # Merge the fast-tracked and newly-fit results.
                if len(fits) == 2:
                    astrom = merge_tables(fits)
                    ref = merge_tables(refs)
                else:
                    astrom = fits[0]
                    ref = refs[0]

                ref.rename('gaia_ra',  'ra')
                ref.rename('gaia_dec', 'dec')

                astrom.raoff  = (ref.ra  - astrom.ra_fit ) * 3600. * np.cos(np.deg2rad(ref.dec))
                astrom.decoff = (ref.dec - astrom.dec_fit) * 3600.

                ok, = np.nonzero(astrom.flux > 0)
                astrom.psfmag = np.zeros(len(astrom), np.float32)
                astrom.psfmag[ok] = -2.5*np.log10(astrom.flux[ok] / exptime) + zp0
                astrom.dpsfmag = np.zeros_like(astrom.psfmag)
                astrom.dpsfmag[ok] = np.abs((-2.5 / np.log(10.)) * astrom.dflux[ok] / astrom.flux[ok])

                dmagall = ref.legacy_survey_mag - astrom.psfmag
                dmag, _, _ = sigmaclip(dmagall, low=2.5, high=2.5)
                ndmag = len(dmag)
                dmagmed = np.median(dmag)
                dmagsig = np.std(dmag)
                zptmed = zp0 + dmagmed
                transp = 10.**(-0.4 * (zp0 - zptmed - kext * (airmass - 1.0)))

                raoff = np.median(astrom.raoff)
                decoff = np.median(astrom.decoff)
        
                print('Tractor PsfEx-fitting results for PS1/Gaia:')
                print('RA, Dec offsets (arcsec) relative to Gaia: %.4f, %.4f' %
                      (raoff, decoff))
                print('RA, Dec stddev (arcsec): %.4f, %.4f' %
                      (np.std(astrom.raoff), np.std(astrom.decoff)))
                print('Mag offset: %.4f' % dmagmed)
                print('Scatter: %.4f' % dmagsig)
                print('Number stars used for zeropoint median %d' % ndmag)
                print('Zeropoint %.4f' % zptmed)
                print('Transparency %.4f' % transp)

                for c in ['x0','y0','x1','y1','flux','raoff','decoff',
                          'dflux','dx','dy']:
                    astrom.set(c, astrom.get(c).astype(np.float32))

                astrom.ra_gaia  = ref.ra
                astrom.dec_gaia = ref.dec
                astrom.phot_g_mean_mag = ref.phot_g_mean_mag
                # Save CCD-level information in the per-star table.
                astrom.ccd_raoff  = np.zeros(len(astrom), np.float32) + raoff
                astrom.ccd_decoff = np.zeros(len(astrom), np.float32) + decoff
                astrom.expnum = np.zeros(len(astrom), np.int32) + self.expnum
                astrom.ccdname = np.array([self.ccdname] * len(astrom))

                # Convert to astropy Table
                cols = astrom.get_columns()
                stars_astrom = Table([astrom.get(c) for c in cols], names=cols)

                # Update the zeropoints table
                ccds['raoff']  = raoff
                ccds['decoff'] = decoff
                ccds['rastddev'] = np.std(astrom.raoff)
                ccds['decstddev'] = np.std(astrom.decoff)
                ra_clip, _, _ = sigmaclip(astrom.raoff, low=3., high=3.)
                ccds['rarms'] = getrms(ra_clip)
                dec_clip, _, _ = sigmaclip(astrom.decoff, low=3., high=3.)
                ccds['decrms'] = getrms(dec_clip)
                ccds['nmatch_astrom'] = len(astrom)

        t0= ptime('all-computations-for-this-ccd',t0)
        # Plots for comparing to Arjuns zeropoints*.ps
        if self.verboseplots:
            self.make_plots(stars,dmag,ccds['zpt'],ccds['transp'])
            t0= ptime('made-plots',t0)
        return ccds, stars_photom, stars_astrom

    def get_splinesky(self):
        # Find splinesky model file and read it
        import tractor
        from tractor.utils import get_class_from_name

        expstr = '%08i' % self.expnum
        # Look for merged file
        fn = os.path.join(self.calibdir, self.camera, 'splinesky-merged', expstr[:5],
                          '%s-%s.fits' % (self.camera, expstr))
        print('Looking for file', fn)
        if os.path.exists(fn):
            T = fits_table(fn)
            I, = np.nonzero((T.expnum == self.expnum) *
                            np.array([c.strip() == self.ext for c in T.ccdname]))
            if len(I) == 1:
                Ti = T[I[0]]

                # Remove any padding
                h,w = Ti.gridh, Ti.gridw
                Ti.gridvals = Ti.gridvals[:h, :w]
                Ti.xgrid = Ti.xgrid[:w]
                Ti.ygrid = Ti.ygrid[:h]

                skyclass = Ti.skyclass.strip()
                clazz = get_class_from_name(skyclass)
                fromfits = getattr(clazz, 'from_fits_row')
                sky = fromfits(Ti)
                return sky

        # Look for single-CCD file
        fn = os.path.join(self.calibdir, self.camera, 'splinesky', expstr[:5], expstr,
                          '%s-%s-%s.fits' % (self.camera, expstr, self.ext))
        print('Reading file', fn)
        if not os.path.exists(fn):
            return None
        
        hdr = fitsio.read_header(fn)
        try:
            skyclass = hdr['SKY']
        except NameError:
            raise NameError('SKY not in header: skyfn=%s, imgfn=%s' % (fn,self.imgfn))
        clazz = get_class_from_name(skyclass)

        if getattr(clazz, 'from_fits', None) is not None:
            fromfits = getattr(clazz, 'from_fits')
            sky = fromfits(fn, hdr)
        else:
            fromfits = getattr(clazz, 'fromFitsHeader')
            sky = fromfits(hdr, prefix='SKY_')

        return sky

    def tractor_fit_sources(self, ref_ra, ref_dec, ref_flux, img, ierr,
                            psf, normalize_psf=True):
        import tractor
        plots = False
        if plots:
            from astrometry.util.plotutils import PlotSequence
            ps = PlotSequence('astromfit')

        print('Fitting positions & fluxes of %i stars' % len(ref_ra))

        cal = fits_table()
        # These x0,y0,x1,y1 are zero-indexed coords.
        cal.x0 = []
        cal.y0 = []
        cal.x1 = []
        cal.y1 = []
        cal.flux = []
        cal.dx = []
        cal.dy = []
        cal.dflux = []
        cal.psfsum = []
        cal.iref = []

        for istar in range(len(ref_ra)):
            ok,x,y = self.wcs.radec2pixelxy(ref_ra[istar], ref_dec[istar])
            x -= 1
            y -= 1
            # Fitting radius
            R = 10
            H,W = img.shape
            xlo = int(x - R)
            ylo = int(y - R)
            if xlo < 0 or ylo < 0:
                continue
            xhi = xlo + R*2
            yhi = ylo + R*2
            if xhi >= W or yhi >= H:
                continue
            subimg = img[ylo:yhi+1, xlo:xhi+1]
            # FIXME -- check that ierr is correct
            subie = ierr[ylo:yhi+1, xlo:xhi+1]
            subpsf = psf.constantPsfAt(x, y)
            psfsum = np.sum(subpsf.img)
            if normalize_psf:
                # print('Normalizing PsfEx model with sum:', s)
                subpsf.img /= psfsum

            #print('PSF model:', subpsf)
            #print('PSF image sum:', subpsf.img.sum())

            tim = tractor.Image(data=subimg, inverr=subie, psf=subpsf)
            flux0 = ref_flux[istar]
            #print('Zp0', zp0, 'mag', ref.mag[istar], 'flux', flux0)
            x0 = x - xlo
            y0 = y - ylo
            src = tractor.PointSource(tractor.PixPos(x0, y0),
                                      tractor.Flux(flux0))
            tr = tractor.Tractor([tim], [src])
            tr.freezeParam('images')
            optargs = dict(priors=False, shared_params=False)
            # The initial flux estimate doesn't seem to work too well,
            # so just for plotting's sake, fit flux first
            src.freezeParam('pos')
            tr.optimize(**optargs)
            src.thawParam('pos')
            #print('Optimizing position of Gaia star', istar)

            if plots:
                plt.clf()
                plt.subplot(2,2,1)
                plt.imshow(subimg, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.subplot(2,2,2)
                mod = tr.getModelImage(0)
                plt.imshow(mod, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow((subimg - mod) * subie, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.suptitle('Before')
                ps.savefig()

            #print('Initial flux', flux0)
            for step in range(50):
                dlnp, x, alpha = tr.optimize(**optargs)
                #print('delta position', src.pos.x - x0, src.pos.y - y0,
                #      'flux', src.brightness, 'dlnp', dlnp)
                if dlnp == 0:
                    break

            #print('Getting variance estimate: thawed params:')
            #tr.printThawedParams()
            variance = tr.optimize(variance=True, just_variance=True, **optargs)
            # Yuck -- if inverse-variance is all zero, weird-shaped result...
            if len(variance) == 4 and variance[3] is None:
                print('No variance estimate available')
                continue

            cal.psfsum.append(psfsum)
            cal.x0.append(x0 + xlo)
            cal.y0.append(y0 + ylo)
            cal.x1.append(src.pos.x + xlo)
            cal.y1.append(src.pos.y + ylo)
            cal.flux.append(src.brightness.getValue())
            cal.iref.append(istar)

            std = np.sqrt(variance)
            cal.dx.append(std[0])
            cal.dy.append(std[1])
            cal.dflux.append(std[2])

            if plots:
                plt.clf()
                plt.subplot(2,2,1)
                plt.imshow(subimg, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.subplot(2,2,2)
                mod = tr.getModelImage(0)
                plt.imshow(mod, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow((subimg - mod) * subie, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.suptitle('After')
                ps.savefig()

        cal.to_np_arrays()
        cal.ra_fit,cal.dec_fit = self.wcs.pixelxy2radec(cal.x1 + 1, cal.y1 + 1)
        return cal

    def get_psfex_model(self):
        import tractor

        expstr = '%08i' % self.expnum
        # Look for merged PsfEx file
        fn = os.path.join(self.calibdir, self.camera, 'psfex-merged', expstr[:5],
                          '%s-%s.fits' % (self.camera, expstr))
        print('Looking for PsfEx file', fn)
        if os.path.exists(fn):
            T = fits_table(fn)
            I, = np.nonzero((T.expnum == self.expnum) *
                            np.array([c.strip() == self.ext for c in T.ccdname]))
            if len(I) == 1:
                Ti = T[I[0]]
                # Remove any padding
                degree = Ti.poldeg1
                # number of terms in polynomial
                ne = (degree + 1) * (degree + 2) // 2
                Ti.psf_mask = Ti.psf_mask[:ne, :Ti.psfaxis1, :Ti.psfaxis2]
                psfex = tractor.PsfExModel(Ti=Ti)
                psf = tractor.PixelizedPsfEx(None, psfex=psfex)
                psf.fwhm = Ti.psf_fwhm
                return psf

        # Look for single-CCD PsfEx file
        fn = os.path.join(self.calibdir, self.camera, 'psfex', expstr[:5], expstr,
                          '%s-%s-%s.fits' % (self.camera, expstr, self.ext))
        print('Reading PsfEx file', fn)
        if not os.path.exists(fn):
            return None
        psf = tractor.PixelizedPsfEx(fn)

        import fitsio
        hdr = fitsio.read_header(fn, ext=1)
        psf.fwhm = hdr['PSF_FWHM']
        return psf
    
    def do_Photometry(self, obj,ps1, ccds,
                      save_xy=False):
        """Measure zeropoint relative to PS1
        
        Args:
            obj: ps1-matched sources detected with dao phot
            ps1: ps1 source matched to obj
            ccds: partially filled _ccds_table
            save_xy: if True save a fits table containing 
                ps1_mag and apmag for matches sources and associated
                photometric cuts

        Returns:
            stars_photom: fits table for stars
            err_message: '' if okay, 'some error text' otherwise, this will end up being
                stored in ccds['err_message']
        """
        print('Photometry on %s stars' % len(ps1))
        objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        cuts,phot= self.get_photometric_cuts(obj,cuts_only=False)
        assert(len(phot['apflux']) == len(obj))
        final_cut= ((cuts['good_flux_and_mag']) &
                    (cuts['no_badpix_in_ap_0']) &
                    (cuts['is_iso']))
        if len(obj[final_cut]) == 0:
            return _stars_table(),'photometry failed, no stars after cuts'

        # Stars table
        ccds['nmatch_photom'] = len(obj[final_cut])
        print('Photometry %s stars after obj cuts' % ccds['nmatch_photom'])

        stars_photom = _stars_table(nstars= ccds['nmatch_photom'])
        stars_photom['apmag'] = phot['apmags'][final_cut]
        stars_photom['ps1_mag'] = ps1.legacy_survey_mag[final_cut]

        if save_xy:
            # Save ps1_mag and apmag for every matched source
            all_stars=fits_table()
            all_stars.set('apmag', phot['apmags'].data)
            all_stars.set('ps1_mag', ps1.legacy_survey_mag)
            all_stars.set('match_x', obj['xcentroid'].data)
            all_stars.set('match_y', obj['ycentroid'].data)
            all_stars.set('match_ra', objra)
            all_stars.set('match_dec', objdec)
            # Then bool cuts for the above arrays
            for key in cuts.keys():
                all_stars.set(key, cuts[key])
            # Avoid memoryview write error
            for col in all_stars.get_columns(): 
                all_stars.set(col,np.array(all_stars.get(col)))
            all_stars.writeto('%s_%s_all_stars.fits' % 
                  (os.path.basename(self.fn).replace('.fits','').replace('.fz',''),
                   self.ccdname))

        # Add additional info
        stars_photom['nmatch']= ccds['nmatch_photom']
        self.add_ccd_info_to_stars_table(stars_photom,
                                         ccds)
        star_kwargs= {"keep": final_cut,
                      "obj":obj,
                      "objra":objra,
                      "objdec":objdec,
                      "apflux":phot['apflux'],
                      "apskyflux":phot['apskyflux'],
                      "apskyflux_perpix":phot['apskyflux_perpix']}
        self.add_obj_info_to_stars_table(stars_photom,**star_kwargs)
        for ps1_band,ps1_iband in zip(['g','r','i','z'],[0,1,2,3]):
            stars_photom['ps1_%s' % ps1_band]= ps1.median[final_cut, ps1_iband]       
        # Zeropoint
        stars_photom['dmagall'] = stars_photom['ps1_mag'] - stars_photom['apmag']
        dmag, _, _ = sigmaclip(stars_photom['dmagall'], low=2.5, high=2.5)
        dmagmed = np.median(dmag)
        dmagsig = np.std(dmag)  # agrees with IDL codes, they just compute std
        zp0 = self.zeropoint(self.band)
        kext = self.extinction(self.band)
        zptmed = zp0 + dmagmed
        transp = 10.**(-0.4 * (zp0 - zptmed - kext * (self.airmass - 1.0)))
        ccds['phoff'] = dmagmed
        ccds['phrms'] = dmagsig
        ccds['zpt'] = zptmed
        ccds['transp'] = transp 

        # Badpix 5 test
        if self.camera in ['90prime','mosaic']:
            # good sources but treat badpix=5 as OK
            final_cut= ((cuts['good_flux_and_mag']) &
                      (cuts['no_badpix_in_ap_0_5']) &
                      (cuts['is_iso']))
            dmagall= ps1.legacy_survey_mag[final_cut] - phot['apmags'][final_cut]
            dmag, _, _ = sigmaclip(dmagall, low=2.5, high=2.5)
            dmagmed = np.median(dmag)
            zp0 = self.zeropoint(self.band)
            kext = self.extinction(self.band)
            zptmed = zp0 + dmagmed
            ccds['zpt_wbadpix5'] = zptmed

        # star,empty string tuple if succeeded
        return stars_photom,''

    def do_Astrometry(self, obj,ref_ra,ref_dec, ccds):
        """Measure ra,dec offsets from Gaia or PS1
       
        Args:
            obj: ps1-matched sources detected with dao phot
            ref_ra,ref_dec: ra and dec of ther ps1 or gaia sources matched to obj
            ccds: partially filled _ccds_table
        
        Returns:
            stars_astrom: fits table for stars
            err_message: '' if okay, 'some error text' otherwise, this will end up being
                stored in ccds['err_message']
        """
        print('Astrometry on %s stars' % len(obj))
        # Cut to obj with good photometry
        cuts= self.get_photometric_cuts(obj,cuts_only=True)
        final_cut= ((cuts['good_flux_and_mag']) &
                    (cuts['no_badpix_in_ap_0']) &
                    (cuts['is_iso']))
        if len(obj[final_cut]) == 0:
            return _stars_table(),'Astromety failed, no stars after cuts'

        ccds['nmatch_astrom'] = len(obj[final_cut])
        print('Astrometry: matched %s sources within %.1f arcsec' % 
              (ccds['nmatch_astrom'], self.match_radius))
       
        # Initialize
        stars_astrom = _stars_table(nstars= ccds['nmatch_astrom'])
        stars_astrom['nmatch']= ccds['nmatch_astrom']
        self.add_ccd_info_to_stars_table(stars_astrom,
                                         ccds)
        # Fill
        objra, objdec = self.wcs.pixelxy2radec(obj[final_cut]['xcentroid']+1, 
                                               obj[final_cut]['ycentroid']+1)
        stars_astrom['radiff'] = (ref_ra[final_cut] - objra) * \
                                  np.cos(np.deg2rad( objdec )) * 3600.0
        stars_astrom['decdiff'] = (ref_dec[final_cut] - objdec) * 3600.0
        
        ccds['raoff'] = np.median(stars_astrom['radiff'])
        ccds['decoff'] = np.median(stars_astrom['decdiff'])
        ccds['rastddev'] = np.std(stars_astrom['radiff'])
        ccds['decstddev'] = np.std(stars_astrom['decdiff'])
        ra_clip, _, _ = sigmaclip(stars_astrom['radiff'], low=3., high=3.)
        ccds['rarms'] = getrms(ra_clip)
        dec_clip, _, _ = sigmaclip(stars_astrom['decdiff'], low=3., high=3.)
        ccds['decrms'] = getrms(dec_clip)
        return stars_astrom,''

    def get_photometric_cuts(self,obj,cuts_only):
        """Do aperture photometry and create a photometric cut base on those measurements
       
        Args:
            obj: sources detected with dao phot
            cuts_only: the final photometric cut will be returned in either case
                True to not compute extra things

        Returns:
            two dicts, cuts and phot
            cuts: keys are ['good_flux_and_mag',
                          'no_badpix_in_ap_0','no_badpix_in_ap_0_5','is_iso']
            phot: keys are ["apflux","apmags","apskyflux","apskyflux_perpix"]
        """
        print('Performing aperture photometry')
        cuts,phot= {},{}

        ap = CircularAperture((obj['xcentroid'], obj['ycentroid']), self.aprad / self.pixscale)
        if self.aper_sky_sub:
            print('**WARNING** using sky apertures for local sky subtraction')
            skyap = CircularAnnulus((obj['xcentroid'], obj['ycentroid']),
                                    r_in=self.skyrad[0] / self.pixscale, 
                                    r_out=self.skyrad[1] / self.pixscale)
            # Use skyap to subtractr local sky
            apphot = aperture_photometry(img, ap)
            #skyphot = aperture_photometry(img, skyap)
            skyphot = aperture_photometry(img, skyap, mask= img_mask > 0)
            apskyflux= skyphot['aperture_sum'] / skyap.area() * ap.area()
            apskyflux_perpix= skyphot['aperture_sum'] / skyap.area() 
            apflux = apphot['aperture_sum'] - apskyflux
        else:
            # ON image not sky subtracted image
            apphot = aperture_photometry(self.img, ap)
            apflux = apphot['aperture_sum']
            # Placeholders
            #apskyflux= apflux.copy()
            #apskyflux.fill(0.)
            #apskyflux_perpix= apskyflux.copy()
        # Get close enough sky/pixel in sky annulus
        # Take cutout of size ~ rout x rout, use same pixels in this slice for sky level
        rin,rout= self.skyrad[0]/self.pixscale, self.skyrad[1]/self.pixscale
        rad= int(np.ceil(rout)) #
        box= 2*rad + 1 # Odd integer so source exactly in center
        use_for_sky= np.zeros((box,box),bool)
        x,y= np.meshgrid(range(box),range(box)) # array valus are the indices
        ind_of_center= rad
        r= np.sqrt((x - ind_of_center)**2 + (y - ind_of_center)**2)
        use_for_sky[(r > rin)*(r <= rout)]= True
        # Get cutout around each source
        apskyflux,apskyflux_perpix=[],[]
        for x,y in zip(obj['xcentroid'].data,obj['ycentroid'].data):
            xc,yc= int(x),int(y)
            x_sl= slice(xc-rad,xc+rad+1)
            y_sl= slice(yc-rad,yc+rad+1)
            cutout= self.img[y_sl,x_sl]
            assert(cutout.shape == use_for_sky.shape)
            from astropy.stats import sigma_clipped_stats
            mean, median, std = sigma_clipped_stats(cutout[use_for_sky], sigma=3.0, iters=5)
            mode_est= 3*median - 2*mean
            apskyflux_perpix.append( mode_est )
        apskyflux_perpix = np.array(apskyflux_perpix) # cnts / pixel
        apskyflux= apskyflux_perpix * ap.area() # cnts / 7'' aperture

        # Aperture flux, mags
        apflux= apflux - apskyflux
        zp0 = self.zeropoint(self.band)
        apmags= - 2.5 * np.log10(apflux.data) + zp0 + 2.5 * np.log10(self.exptime)
        
        #obj = obj[istar]
        #if len(obj) == 0:
        #    print('No sources away from edges, crash')
        #    return ccds, _stars_table()

        # Remove stars if saturated within 5 pixels of centroid
        ap_for_mask = CircularAperture((obj['xcentroid'], obj['ycentroid']), 5.)

        img_mask= self.get_zero_one_mask(self.bitmask)
        phot_for_mask = aperture_photometry(img_mask, ap_for_mask)
        flux_for_mask = phot_for_mask['aperture_sum'] 
        
        # Additional mask for testing
        img_mask_5= self.get_zero_one_mask(self.bitmask,good=[5])
        phot_for_mask_5 = aperture_photometry(img_mask_5, ap_for_mask)
        flux_for_mask_5 = phot_for_mask_5['aperture_sum'] 
        
        # No stars within our skyrad_outer (10'')
        nn_sep= self.skyrad[0] 
        objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        b_isolated= self.isolated_radec(objra,objdec,nn=2,minsep= nn_sep/3600.)
      
        # 2nd round of cuts:  
        # In order of biggest affect: isolated,apmags, apflux, flux_for_mask
        cuts['good_flux_and_mag']= ((apflux.data > 0) &
                                    (apmags > 12.) &
                                    (apmags < 30.))
        cuts['no_badpix_in_ap_0']= flux_for_mask.data == 0
        cuts['no_badpix_in_ap_0_5']= flux_for_mask_5.data == 0
        cuts['is_iso']= b_isolated
        
        print('2nd round of cuts')
        print("flux > 0 & 12 < mag < 30: %d/%d" % (len(obj[cuts['good_flux_and_mag']]),len(obj))  )
        print("no masked pixels in ap: %d/%d" % (len(obj[cuts['no_badpix_in_ap_0']]),len(obj)) )
        print("isolated source: %d/%d" % (len(obj[cuts['is_iso']]),len(obj)) )
      
        if cuts_only:
            return cuts
        else:
            phot={"apflux":apflux,
                "apmags":apmags,
                "apskyflux":apskyflux,
                "apskyflux_perpix":apskyflux_perpix}
            return cuts, phot

    def add_ccd_info_to_stars_table(self,stars,ccds):
        """Adds info to stars table that is inferable from ccd header

        Args:
            stars: the stars table
            ccds: ccds table
        """
        stars['image_filename'] =ccds['image_filename']
        stars['image_hdu']= ccds['image_hdu'] 
        stars['expnum'] = self.expnum
        stars['expid'] = self.expid
        stars['filter'] = self.band
        stars['ccdname'] = self.ccdname
        stars['gain'] = self.gain
        stars['exptime'] = self.exptime
 
    def add_obj_info_to_stars_table(self,stars,
                                    keep,obj,
                                    objra,objdec,
                                    apflux,apskyflux,apskyflux_perpix):
        """Adds arrays from obj to stars table

        Args:
            stars: the stars table
            keep: bool array of obj indices to include in stars table 
            obj: 
            objra,objdec,apflux,apskyflux,apskyflux_perpix:
        """
        assert(len(stars) == len(obj[keep]))
        stars['x'] = obj['xcentroid'][keep]
        stars['y'] = obj['ycentroid'][keep]
        stars['ra'] = objra[keep]
        stars['dec'] = objdec[keep]
        stars['apflux'] = apflux[keep]
        stars['apskyflux'] = apskyflux[keep]
        stars['apskyflux_perpix'] = apskyflux_perpix[keep]

    def make_plots(self,stars,dmag,zpt,transp):
        '''stars -- stars table'''
        suffix='_qa_%s.png' % stars['expid'][0][-4:]
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        plt.subplots_adjust(wspace=0.2,bottom=0.2,right=0.8)
        for key in ['astrom_gaia','photom']:
        #for key in ['astrom_gaia','astrom_ps1','photom']:
            if key == 'astrom_gaia':    
                ax[0].scatter(stars['radiff'],stars['decdiff'])
                xlab=ax[0].set_xlabel(r'$\Delta Ra$ (Gaia - CCD)')
                ylab=ax[0].set_ylabel(r'$\Delta Dec$ (Gaia - CCD)')
            elif key == 'astrom_ps1':  
                raise ValueError('not needed')  
                ax.scatter(stars['radiff_ps1'],stars['decdiff_ps1'])
                ax.set_xlabel(r'$\Delta Ra [arcsec]$ (PS1 - CCD)')
                ax.set_ylabel(r'$\Delta Dec [arcsec]$ (PS1 - CCD)')
                ax.text(0.02, 0.95,'Median: %.4f,%.4f' % \
                          (np.median(stars['radiff_ps1']),np.median(stars['decdiff_ps1'])),\
                        va='center',ha='left',transform=ax.transAxes,fontsize=20)
                ax.text(0.02, 0.85,'RMS: %.4f,%.4f' % \
                          (getrms(stars['radiff_ps1']),getrms(stars['decdiff_ps1'])),\
                        va='center',ha='left',transform=ax.transAxes,fontsize=20)
            elif key == 'photom':
                ax[1].hist(dmag)
                xlab=ax[1].set_xlabel('PS1 - AP mag (main seq, 2.5 clipping)')
                ylab=ax[1].set_ylabel('Number of Stars')
        # List key numbers
        ax[1].text(1.02, 1.,r'$\Delta$ Ra,Dec',\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=12)
        ax[1].text(1.02, 0.9,r'  Median: %.4f,%.4f' % \
                  (np.median(stars['radiff']),np.median(stars['decdiff'])),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.80,'  RMS: %.4f,%.4f' % \
                  (getrms(stars['radiff']),getrms(stars['decdiff'])),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.7,'PS1-CCD Mag',\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=12)
        ax[1].text(1.02, 0.6,'  Median:%.4f,%.4f' % \
                  (np.median(dmag),np.std(dmag)),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.5,'  Stars: %d' % len(dmag),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.4,'  Zpt=%.4f' % zpt,\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.3,'  Transp=%.4f' % transp,\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        # Save
        fn= self.zptsfile.replace('.fits',suffix)
        plt.savefig(fn,bbox_extra_artists=[xlab,ylab])
        plt.close()
        print('Wrote %s' % fn)

    def run_calibs(self, ext):
        self.set_hdu(ext)
        psfex = False
        splinesky = False
        if self.get_psfex_model() is None:
            psfex = True
        if self.get_splinesky() is None:
            splinesky = True
        if (not psfex) and (not splinesky):
            # Nothing to do!
            return

        from legacypipe.survey import LegacySurveyData
        from legacypipe.decam import DecamImage

        class FakeLegacySurveyData(LegacySurveyData):
            def get_calib_dir(self):
                return self.calibdir

        class FakeCCD(object):
            pass

        survey = FakeLegacySurveyData()
        survey.calibdir = self.calibdir
        ccd = FakeCCD()
        ccd.image_filename = self.fn
        ccd.image_hdu = self.image_hdu
        ccd.expnum = self.expnum
        ccd.ccdname = self.ccdname
        ccd.filter = self.band
        ccd.exptime = self.exptime
        ccd.camera = self.camera
        ccd.ccdzpt = 25.0
        ccd.ccdraoff = 0.
        ccd.ccddecoff = 0.
        ccd.fwhm = 0.
        ccd.propid = self.propid
        # fake
        ccd.cd1_1 = ccd.cd2_2 = self.pixscale / 3600.
        ccd.cd1_2 = ccd.cd2_1 = 0.
        ccd.pixscale = self.pixscale ## units??
        ccd.mjd_obs = self.mjd_obs
        ccd.width = 0
        ccd.height = 0
        ccd.arawgain = self.gain
        
        im = survey.get_image_object(ccd)
        im.run_calibs(psfex=psfex, sky=splinesky, splinesky=True)


class DecamMeasurer(Measurer):
    '''DECam CP units: ADU
    Class to measure a variety of quantities from a single DECam CCD.

    Image read will be converted to e-
    also zpt to e-
    '''
    def __init__(self, *args, **kwargs):
        super(DecamMeasurer, self).__init__(*args, **kwargs)

        self.minstar= 5 # decstat.pro
        self.pixscale=0.262 
        self.camera = 'decam'
        self.ut = self.primhdr['TIME-OBS']
        self.band = self.get_band()
        # {RA,DEC}: center of exposure, TEL{RA,DEC}: boresight of telescope
        # Use center of exposure if possible
        if 'RA' in self.primhdr.keys():
            self.ra_bore = self.primhdr['RA']
            self.dec_bore = self.primhdr['DEC']
        elif 'TELRA' in self.primhdr.keys():
            self.ra_bore = self.primhdr['TELRA']
            self.dec_bore = self.primhdr['TELDEC']
        else:
            raise ValueError('Neither RA or TELRA in pimhdr, crash')
        if type(self.ra_bore) == str:
            self.ra_bore = hmsstring2ra(self.ra_bore) 
            self.dec_bore = dmsstring2dec(self.dec_bore)
        #self.gain = self.hdr['ARAWGAIN'] # hack! average gain [electron/sec]

        # /global/homes/a/arjundey/idl/pro/observing/decstat.pro
        self.zp0 =  dict(g = 26.610,r = 26.818,z = 26.484,
                         # i,Y from DESY1_Stripe82 95th percentiles
                         i=26.758, Y=25.321) # e/sec
        self.sky0 = dict(g = 22.04,r = 20.91,z = 18.46,
                         # i, Y totally made up
                         i=19.68, Y=18.46) # AB mag/arcsec^2
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06,
                          #i, Y totally made up
                          i=0.08, Y=0.06)
        # --> e/sec
        #for b in self.zp0.keys(): 
        #    self.zp0[b] += -2.5*np.log10(self.gain) 
        # Dict: {"ccd col":[possible CP Header keys for that]}
        self.cp_header_keys= {'width':['ZNAXIS1','NAXIS1'],
                              'height':['ZNAXIS2','NAXIS2'],
                              'fwhm_cp':['FWHM']}

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band

    def get_gain(self,hdr):
        return np.average((hdr['GAINA'],hdr['GAINB']))
        #return hdr['ARAWGAIN']

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacyanalysis.ps1cat import ps1_to_decam
        return ps1_to_decam(ps1stars, band)

    def scale_image(self, img):
        return img * self.gain

    def scale_weight(self, img):
        return img / (self.gain**2)

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion
    
class Mosaic3Measurer(Measurer):
    '''Class to measure a variety of quantities from a single Mosaic3 CCD.
    UNITS: e-/s'''
    def __init__(self, *args, **kwargs):
        super(Mosaic3Measurer, self).__init__(*args, **kwargs)

        self.minstar=20 # mosstat.pro
        self.pixscale=0.262 # 0.260 is right, but mosstat.pro has 0.262
        self.camera = 'mosaic'
        self.band= self.get_band()
        self.ut = self.primhdr['TIME-OBS']
        # {RA,DEC}: center of exposure, TEL{RA,DEC}: boresight of telescope
        self.ra_bore = hmsstring2ra(self.primhdr['RA'])
        self.dec_bore = dmsstring2dec(self.primhdr['DEC'])
        # ARAWGAIN does not exist, 1.8 or 1.94 close
        #self.gain = self.hdr['GAIN']

        self.zp0 = dict(z = 26.552)
        self.sky0 = dict(z = 18.46)
        self.k_ext = dict(z = 0.06)
        # --> e/sec
        #for b in self.zp0.keys(): 
        #    self.zp0[b] += -2.5*np.log10(self.gain)  
        # Dict: {"ccd col":[possible CP Header keys for that]}
        self.cp_header_keys= {'width':['ZNAXIS1','NAXIS1'],
                              'height':['ZNAXIS2','NAXIS2'],
                              'fwhm_cp':['SEEINGP1','SEEINGP']}
   
    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0][0] # zd --> z
        return band

    def get_gain(self,hdr):
        return hdr['GAIN']
        #return np.average((hdr['GAINA'],hdr['GAINB']))
        #return hdr['ARAWGAIN']

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacyanalysis.ps1cat import ps1_to_mosaic
        return ps1_to_mosaic(ps1stars, band)

    def scale_image(self, img):
        '''Convert image from electrons/sec to electrons.'''
        return img * self.exptime

    def scale_weight(self, img):
        return img / (self.exptime**2)

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion

class NinetyPrimeMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single 90prime CCD.
    UNITS -- CP e-/s'''
    def __init__(self, *args, **kwargs):
        super(NinetyPrimeMeasurer, self).__init__(*args, **kwargs)
        
        self.minstar=20 # bokstat.pro
        self.pixscale= 0.470 # 0.455 is correct, but mosstat.pro has 0.470
        self.camera = '90prime'
        self.band= self.get_band()
        # {RA,DEC}: center of exposure, doesn't have TEL{RA,DEC}
        self.ra_bore = hmsstring2ra(self.primhdr['RA'])
        self.dec_bore = dmsstring2dec(self.primhdr['DEC'])
        self.ut = self.primhdr['UT']

        # Can't find what people are using for this!
        # 1.4 is close to average of hdr['GAIN[1-16]']
        #self.gain= 1.4 
        
        #self.gain = np.average((self.hdr['GAINA'],self.hdr['GAINB'])) 
        # Average (nominal) gain values.  The gain is sort of a hack since this
        # information should be scraped from the headers, plus we're ignoring
        # the gain variations across amplifiers (on a given CCD).
        #gaindict = dict(ccd1 = 1.47, ccd2 = 1.48, ccd3 = 1.42, ccd4 = 1.4275)
        #self.gain = gaindict[self.ccdname.lower()]

        # Nominal zeropoints, sky brightness, and extinction values (taken from
        # rapala.ninetyprime.boketc.py).  The sky and zeropoints are both in
        # ADU, so account for the gain here.
        #corr = 2.5 * np.log10(self.gain)

        # /global/homes/a/arjundey/idl/pro/observing/bokstat.pro
        self.zp0 =  dict(g = 26.93,r = 27.01,z = 26.552) # ADU/sec
        self.sky0 = dict(g = 22.04,r = 20.91,z = 18.46) # AB mag/arcsec^2
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06)
        # --> e/sec
        #for b in self.zp0.keys(): 
        #    self.zp0[b] += -2.5*np.log10(self.gain)  
        # Dict: {"ccd col":[possible CP Header keys for that]}
        self.cp_header_keys= {'width':['ZNAXIS1','NAXIS1'],
                              'height':['ZNAXIS2','NAXIS2'],
                              'fwhm_cp':['SEEINGP1','SEEINGP']}
    
    def get_gain(self,hdr):
        return 1.4 # no GAINA,B

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band.replace('bokr', 'r')

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacyanalysis.ps1cat import ps1_to_90prime
        return ps1_to_90prime(ps1stars, band)

    def scale_image(self, img):
        '''Convert image from electrons/sec to electrons.'''
        return img * self.exptime

    def scale_weight(self, img):
        return img / (self.exptime**2)

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion


def get_extlist(camera,fn,debug=False,choose_ccd=None):
    '''
    Args:
        fn: image fn to read hdu from
        debug: use subset of the ccds
        choose_ccd: if not None, use only this ccd given

    Returns: 
        list of hdu names 
    '''
    if camera == '90prime':
        extlist = ['CCD1', 'CCD2', 'CCD3', 'CCD4']
        if debug:
            extlist = ['CCD1']
            #extlist = ['CCD1','CCD2']
    elif camera == 'mosaic':
        extlist = ['CCD1', 'CCD2', 'CCD3', 'CCD4']
        if debug:
            extlist = ['CCD2']
    elif camera == 'decam':
        hdu= fitsio.FITS(fn)
        extlist= [hdu[i].get_extname() for i in range(1,len(hdu))]
        if debug:
            extlist = ['N4'] #,'S4', 'S22','N19']
    else:
        print('Camera {} not recognized!'.format(camera))
        pdb.set_trace() 
    if choose_ccd:
        print('CHOOSING CCD %s' % choose_ccd)
        extlist= [choose_ccd]
    return extlist
   
 
def _measure_image(args):
    '''Utility function to wrap measure_image function for multiprocessing map.''' 
    return measure_image(*args)

def measure_image(img_fn, run_calibs=False, **measureargs):
    '''Wrapper on the camera-specific classes to measure the CCD-level data on all
    the FITS extensions for a given set of images.
    '''
    t0= Time()

    print('Working on image {}'.format(img_fn))

    # Fitsio can throw error: ValueError: CONTINUE not supported
    try:
        print('img_fn=%s' % img_fn)
        primhdr = fitsio.read_header(img_fn, ext=0)
    except ValueError:
        # astropy can handle it
        tmp= fits_astropy.open(img_fn)
        primhdr= tmp[0].header
        tmp.close()
        del tmp
#        # skip zpt for this image 
#        print('Error reading img_fn=%s, see %s' % \
#                (img_fn,'zpts_bad_headerskipimage.txt')) 
#        with open('zpts_bad_headerskipimage.txt','a') as foo:
#            foo.write('%s\n' % (img_fn,))
#        ccds = []
#        stars = []
#        # FIX ME!! 4 should depend on camera, 60 for decam, 4 for mosaic,bok
#        for cnt in range(4):
#            ccds.append( _ccds_table() )
#            stars.append( _stars_table() )
#        ccds = vstack(ccds)
#        stars = vstack(stars)
#        return ccds,stars
    
    camera= measureargs['camera']
    camera_check = primhdr.get('INSTRUME','').strip().lower()
    # mosaic listed as mosaic3 in header, other combos maybe
    assert(camera in camera_check or camera_check in camera)
    
    extlist = get_extlist(camera,img_fn, 
                          debug=measureargs['debug'],
                          choose_ccd=measureargs['choose_ccd'])
    nnext = len(extlist)

    if camera == 'decam':
        measure = DecamMeasurer(img_fn, **measureargs)
    elif camera == 'mosaic':
        measure = Mosaic3Measurer(img_fn, **measureargs)
    elif camera == '90prime':
        measure = NinetyPrimeMeasurer(img_fn, **measureargs)
    extra_info= dict(zp_fid= measure.zeropoint( measure.band ),
                     sky_fid= measure.sky( measure.band ),
                     ext_fid= measure.extinction( measure.band ),
                     exptime= measure.exptime,
                     pixscale= measure.pixscale)
    
    all_ccds = []
    all_stars_photom = []
    all_stars_astrom = []
    psfex = measureargs['psf']
    splinesky = measureargs['splinesky']
    for ext in extlist:
        if run_calibs:
            measure.run_calibs(ext)

        ccds, stars_photom, stars_astrom = measure.run(ext, psfex=psfex, splinesky=splinesky, save_xy=measureargs['debug'])
        t0= ptime('measured-ext-%s' % ext,t0)

        if ccds is not None:
            all_ccds.append(ccds)
        if stars_photom is not None:
            all_stars_photom.append(stars_photom)
        if stars_astrom is not None:
            all_stars_astrom.append(stars_astrom)

    # Compute the median zeropoint across all the CCDs.
    all_ccds = vstack(all_ccds)
    all_stars_photom = vstack(all_stars_photom)
    all_stars_astrom = vstack(all_stars_astrom)
    zpts = all_ccds['zpt']
    all_ccds['zptavg'] = np.median(zpts[np.isfinite(zpts)])

    t0= ptime('measure-image-%s' % img_fn,t0)
    return all_ccds, all_stars_photom, all_stars_astrom, extra_info


class outputFns(object):
    def __init__(self,imgfn,outdir, not_on_proj=False,
                 copy_from_proj=False, debug=False):
        """Assigns filename, makes needed dirs, and copies images to scratch if needed

        Args:
            imgfn: abs path to image, should be a ooi or oki file
            outdir: root dir for outptus
            not_on_proj: True if image not stored on project or projecta
            copy_from_proj: True if want to copy all image files from project to scratch
            debug: 4 ccds only if true

        Attributes:
            imgfn: image that will be read
            zptfn: zeropoints file
            starfn: stars file

        Example:
            outdir/decam/DECam_CP/CP20151226/img_fn.fits.fz
            outdir/decam/DECam_CP/CP20151226/img_fn-zpt%s.fits
            outdir/decam/DECam_CP/CP20151226/img_fn-star%s.fits
        """
        # img fns
        if not_on_proj:
            # Don't worry about mirroring path, just get image's fn
            self.imgfn= imgfn
            dirname='' 
            basename= os.path.basename(imgfn) 
        else:
            # Mirror path to image but write to scratch
            proj_dir= '/project/projectdirs/cosmo/staging/'
            proja_dir= '/global/projecta/projectdirs/cosmo/staging/'
            assert( (proj_dir in imgfn) |
                  (proja_dir in imgfn))
            relative_fn= imgfn.replace(proj_dir,'').replace(proja_dir,'')
            dirname= os.path.dirname(relative_fn) 
            basename= os.path.basename(relative_fn) 
            if copy_from_proj:
                # somwhere on scratch
                self.imgfn= os.path.join(outdir,dirname,basename)
            else:
                self.imgfn= imgfn
        # zpt,star fns
        self.zptfn= os.path.join(outdir,dirname,
                                 basename.replace('.fits.fz','-zpt.fits')) 
        self.starfn_photom= os.path.join(outdir,dirname,
                                 basename.replace('.fits.fz','-star-photom.fits')) 
        self.starfn_astrom= os.path.join(outdir,dirname,
                                 basename.replace('.fits.fz','-star-astrom.fits')) 
        if debug:
            self.zptfn= self.zptfn.replace('-zpt','-debug-zpt')
            self.starfn_photom= self.starfn_photom.replace('-star','-debug-star')
            self.starfn_astrom= self.starfn_astrom.replace('-star','-debug-star')
        # Makedirs
        try:
            os.makedirs(os.path.join(outdir,dirname))
        except FileExistsError: 
            print('Directory already exists: %s' % os.path.join(outdir,dirname))
        # Copy if need
        if copy_from_proj:
            if not os.path.exists(self.imgfn): 
                dobash("cp %s %s" % (imgfn,self.imgfn))
            if not os.path.exists( get_bitmask_fn(self.imgfn)): 
                dobash("cp %s %s" % ( get_bitmask_fn(imgfn), get_bitmask_fn(self.imgfn)))

            
#def success(ccds,imgfn, debug=False, choose_ccd=None):
#    num_ccds= dict(decam=60,mosaic=4)
#    num_ccds['90prime']=4
#    hdu= fitsio.FITS(imgfn)
#    #if len(ccds) >= num_ccds.get(camera,0):
#    if len(ccds) == len(hdu)-1:
#        return True
#    elif debug and len(ccds) >= 1:
#        # only 1 ccds needs to be done if debuggin
#        return True
#    elif choose_ccd and len(ccds) >= 1:
#        return True
#    else:
#        return False


def runit(imgfn,zptfn,starfn_photom,starfn_astrom,
          **measureargs):
    '''Generate a legacypipe-compatible CCDs file for a given image.
    '''
    t0 = Time()
    ccds, stars_photom, stars_astrom, extra_info= measure_image(imgfn, **measureargs)
    t0= ptime('measure_image',t0)

    # Only write if all CCDs are done
    #if success(ccds,imgfn, 
    #           debug=measureargs['debug'],
    #           choose_ccd=measureargs['choose_ccd']):
    # Write out.
    ccds.write(zptfn)
    # Header <-- fiducial zp,sky,ext, also exptime, pixscale
    hdulist = fits_astropy.open(zptfn, mode='update')
    prihdr = hdulist[0].header
    for key,val in extra_info.items():
        prihdr[key] = val
    hdulist.close() # Save changes
    print('Wrote {}'.format(zptfn))
    # zpt --> Legacypipe table
    create_legacypipe_table(zptfn, camera=measureargs['camera'])
    # Two stars tables
    stars_photom.write(starfn_photom)
    stars_astrom.write(starfn_astrom)
    print('Wrote 2 stars tables\n%s\n%s' %  (starfn_photom,starfn_astrom))
    # Clean up
    t0= ptime('write-results-to-fits',t0)
    if measureargs['copy_from_proj'] & os.path.exists(imgfn): 
        # Safegaurd against removing stuff on /project
        assert(not 'project' in imgfn)
        #dobash("rm %s" % imgfn_scr)
        #dobash("rm %s" % dqfn_scr)
        dobash("SOFT rm %s" % imgfn_scr)
        dobash("SOFT rm %s" % dqfn_scr)
        t0= ptime('removed-cp-from-scratch',t0)
    
def parse_coords(s):
    '''stackoverflow: 
    https://stackoverflow.com/questions/9978880/python-argument-parser-list-of-list-or-tuple-of-tuples'''
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")

def get_parser():
    '''return parser object, tells it what options to look for
    options can come from a list of strings or command line'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                     description='Generate a legacypipe-compatible CCDs file \
                                                  from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--image',action='store',default=None,help='relative path to image starting from decam,bok,mosaicz dir',required=False)
    parser.add_argument('--image_list',action='store',default=None,help='text file listing multiples images in same was as --image',required=False)
    parser.add_argument('--outdir', type=str, default='.', help='Where to write zpts/,images/,logs/')
    parser.add_argument('--not_on_proj', action='store_true', default=False, help='set when the image is not on project or projecta')
    parser.add_argument('--copy_from_proj', action='store_true', default=False, help='copy image data from proj to scratch before analyzing')
    parser.add_argument('--debug', action='store_true', default=False, help='Write additional files and plots for debugging')
    parser.add_argument('--choose_ccd', action='store', default=None, help='forced to use only the specified ccd')
    parser.add_argument('--ps1_only', action='store_true', default=False, help='only ps1 (not gaia) for astrometry. For photometry, only ps1 is used no matter what')
    parser.add_argument('--ps1_pattern', action='store', default='/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-%(hp)05d.fits', help='pattern for PS1 catalogues')
    parser.add_argument('--ps1_gaia_pattern', action='store', default='/project/projectdirs/cosmo/work/gaia/chunks-ps1-gaia/chunk-%(hp)05d.fits', help='pattern for PS1-Gaia Matched-only catalogues')
    parser.add_argument('--det_thresh', type=float, default=10., help='source detection, 10x sky sigma')
    parser.add_argument('--match_radius', type=float, default=1., help='arcsec, matching to gaia/ps1, 1 arcsec better astrometry than 3 arcsec as used by IDL codes')
    parser.add_argument('--sn_min', type=float,default=None, help='min S/N, optional cut on apflux/sqrt(skyflux)')
    parser.add_argument('--sn_max', type=float,default=None, help='max S/N, ditto')
    parser.add_argument('--aper_sky_sub', action='store_true',default=False,
                        help='Changes local sky subtraction step. Do aperture sky subraction instead of subtracting legacypipe splinesky')
    parser.add_argument('--logdir', type=str, default='.', help='Where to write zpts/,images/,logs/')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to prepend to the output files.')
    parser.add_argument('--verboseplots', action='store_true', default=False, help='use to plot FWHM Moffat PSF fits to the 20 brightest stars')
    parser.add_argument('--aprad', type=float, default=3.5, help='Aperture photometry radius (arcsec).')
    parser.add_argument('--skyrad_inner', type=float, default=7.0, help='Radius of inner sky annulus (arcsec).')
    parser.add_argument('--skyrad_outer', type=float, default=10.0, help='Radius of outer sky annulus (arcsec).')
    parser.add_argument('--calibrate', action='store_true',
                        help='Use this option when deriving the photometric transformation equations.')
    parser.add_argument('--nproc', type=int,action='store',default=1,
                        help='set to > 1 if using legacy-zeropoints-mpiwrapper.py')
    parser.add_argument('--run-calibs', default=False, action='store_true',
                        help='Create PsfEx and splinesky files if they do not already exist')
    parser.add_argument('--psf', default=False, action='store_true',
                        help='Use PsfEx model for astrometry & photometry')
    parser.add_argument('--splinesky', default=False, action='store_true',
                        help='Use spline sky model for sky subtraction?')
    parser.add_argument('--calibdir', default=None, action='store',
                        help='if None will use LEGACY_SURVEY_DIR/calib, e.g. /global/cscratch1/sd/desiproc/dr5-new/calib')
    return parser


def main(image_list=None,args=None): 
    ''' Produce zeropoints for all CP images in image_list
    image_list -- iterable list of image filenames
    args -- parsed argparser objection from get_parser()'''
    assert(not args is None)
    assert(not image_list is None)
    t0 = Time()
    tbegin=t0
    
    # Build a dictionary with the optional inputs.
    measureargs = vars(args)
    measureargs.pop('image_list')
    measureargs.pop('image')
    # Add user specified camera, useful check against primhdr
    #measureargs.update(dict(camera= args.camera))

    outdir = measureargs.pop('outdir')
    try_mkdir(outdir)
    t0=ptime('parse-args',t0)
    for imgfn in image_list:
        # Check if zpt already written
        F= outputFns(imgfn, outdir,
                     not_on_proj= measureargs['not_on_proj'],
                     copy_from_proj= measureargs['copy_from_proj'],
                     debug=measureargs['debug'])
        if (os.path.exists(F.zptfn) & 
            os.path.exists(F.starfn_photom) & 
            os.path.exists(F.starfn_astrom) ):
            print('Already finished: %s' % F.zptfn)
            continue
        # Create the file
        t0=ptime('b4-run',t0)
        runit(F.imgfn,F.zptfn,F.starfn_photom,F.starfn_astrom, 
              **measureargs)
        #try: 
        #except:
        #    print('zpt failed for %s' % imgfn_proj)
        t0=ptime('after-run',t0)
    tnow= Time()
    print("TIMING:total %s" % (tnow-tbegin,))
    print("Done")
   
if __name__ == "__main__":
    parser= get_parser()  
    args = parser.parse_args()
    if args.image_list:
        images= read_lines(args.image_list) 
    elif args.image:
        images= [args.image]

    main(image_list=images,args=args)


