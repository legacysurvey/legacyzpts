from __future__ import print_function
import numpy as np
import pylab as plt
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence
from collections import Counter

def psf_cuts_to_string(ccd_cuts, join=', '):
    s = []
    for k,v in CCD_CUT_BITS.items():
        if ccd_cuts & v:
            s.append(k)
    return join.join(s)

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
    phrms = 0x400,
    radecrms = 0x800,
    seeing_bad = 0x1000,
    early_decam = 0x2000,
    depth_cut = 0x4000,
)

MJD_EARLY_DECAM = 56516.

def dr7_early_decam_update():
    from legacypipe.survey import LegacySurveyData
    survey = LegacySurveyData('/global/cscratch1/sd/desiproc/dr7')
    ccds = survey.get_ccds()
    print('Setting early_decam bit for', np.sum(ccds.mjd_obs < MJD_EARLY_DECAM), 'CCDs')
    ccds.ccd_cuts |= CCD_CUT_BITS['early_decam'] * (ccds.mjd_obs < MJD_EARLY_DECAM)
    ccds.writeto('/tmp/survey-ccds-dr7.fits.gz')

    ccds.cut(ccds.ccd_cuts == 0)
    ccds.writeto('/tmp/ccds.fits')
    cmd = 'startree -i /tmp/ccds.fits -o /tmp/survey-ccds-dr7.kd.fits -P -k -n ccds -T'
    print(cmd)
    import os
    os.system(cmd)

    ann = survey.get_annotated_ccds()
    print('Setting early_decam bit for', np.sum(ann.mjd_obs < MJD_EARLY_DECAM), 'CCDs')
    ann.ccd_cuts |= CCD_CUT_BITS['early_decam'] * (ann.mjd_obs < MJD_EARLY_DECAM)
    ann.writeto('/tmp/ccds-annotated-dr7.fits.gz')


def psf_zeropoint_cuts(P, pixscale,
                       zpt_cut_lo, zpt_cut_hi, bad_expid, camera):
    '''
    zpt_cut_lo, zpt_cut_hi: dict from band to zeropoint.
    '''

    ## PSF zeropoints cuts

    P.ccd_cuts = np.zeros(len(P), np.int32)

    seeing = np.isfinite(P.fwhm) * P.fwhm * pixscale
    P.zpt[np.logical_not(np.isfinite(P.zpt))] = 0.
    P.ccdzpt[np.logical_not(np.isfinite(P.ccdzpt))] = 0.
    P.ccdphrms[np.logical_not(np.isfinite(P.ccdphrms))] = 1.
    P.ccdrarms[np.logical_not(np.isfinite(P.ccdrarms))] = 1.
    P.ccddecrms[np.logical_not(np.isfinite(P.ccddecrms))] = 1.

    keys = zpt_cut_lo.keys()

    cuts = [
        ('not_grz',   np.array([f.strip() not in keys for f in P.filter])),
        ('zpt_small', np.array([ccdzpt < zpt_cut_lo.get(f,0) for f,ccdzpt in zip(P.filter, P.ccdzpt)])),
        ('zpt_large', np.array([ccdzpt > zpt_cut_hi.get(f,0) for f,ccdzpt in zip(P.filter, P.ccdzpt)])),
        ('phrms',     P.ccdphrms > 0.1),
        ('radecrms',  np.logical_or(P.ccdrarms > 0.25,
                                    P.ccddecrms > 0.25)),
        ('exptime_lt_30', P.exptime < 30),
        ('zpt_diff_avg', (np.abs(P.ccdzpt - P.zpt) > 0.25)),
        ('seeing_bad', np.logical_or(seeing < 0, seeing > 3.0)),
        ('badexp_file', np.array([expnum in bad_expid for expnum in P.expnum])),
    ]

    if camera == 'mosaic':
        cuts.append(('not_third_pix', (np.logical_not(P.yshift) * (P.mjd_obs < 57674.))))

    if camera == 'decam':
        cuts.append(('early_decam', P.mjd_obs < MJD_EARLY_DECAM))

    for name,cut in cuts:
        P.ccd_cuts += CCD_CUT_BITS[name] * cut
        print(np.count_nonzero(cut), 'CCDs cut by', name)

def read_bad_expid(fn='bad_expid.txt'):
    bad_expid = {}
    f = open(fn)
    for line in f.readlines():
        #print(line)
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        words = line.split()
        if len(words) < 2:
            continue
        try:
            expnum = int(words[0], 10)
        except:
            print('Skipping line', line)
            continue
        reason = ' '.join(words[1:])
        bad_expid[expnum] = reason
    return bad_expid

if __name__ == '__main__':

    dr7_early_decam_update()
    import sys
    sys.exit(0)


    bad_expid = read_bad_expid()

    g0 = 25.74
    r0 = 25.52
    z0 = 26.20

    dg = (-0.5, 0.18)
    dr = (-0.5, 0.18)
    dz = (-0.6, 0.6)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus5.fits')
    S = psf_zeropoint_cuts(P, 0.262,
                           dict(z=z0+dz[0]), dict(z=z0+dz[1]),
                           bad_expid, 'mosaic')
    S.writeto('survey-ccds-mosaic-dr6plus5.fits')
    sys.exit(0)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus4.fits')
    S = psf_zeropoint_cuts(P, ['z'], 0.262,
                           z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    S.writeto('survey-ccds-mosaic-dr6plus4.fits')
    sys.exit(0)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus3.fits')
    S = psf_zeropoint_cuts(P, ['z'], 0.262,
                           z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    S.writeto('survey-ccds-mosaic-dr6plus3.fits')
    sys.exit(0)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus2.fits')
    S = psf_zeropoint_cuts(P, ['z'], 0.262,
                           z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    S.writeto('survey-ccds-mosaic-dr6plus2.fits')
    sys.exit(0)

    P = fits_table('dr6plus.fits')
    S = psf_zeropoint_cuts(P, ['z'], 0.262,
                           z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    S.writeto('survey-ccds-dr6plus.fits')
    sys.exit(0)
    

    for X in [
            (#'apzpts/survey-ccds-90prime-legacypipe.fits.gz',
                'apzpts/survey-ccds-90prime.fits.gz',
                'survey-ccds-90prime-psfzpts.fits',
                #'90prime-psfzpts.fits',
                'g', 'BASS g', 'g', 20, 25, 26.25, 0.45,
                #25.2, 26.0,
                g0+dg[0], g0+dg[1], {}, '90prime'),
            (#'apzpts/survey-ccds-90prime-legacypipe.fits.gz',
                'apzpts/survey-ccds-90prime.fits.gz',
                'survey-ccds-90prime-psfzpts.fits',
                #'90prime-psfzpts.fits',
                'r', 'BASS r', 'r', 19.5, 24.75, 25.75, 0.45,
                #24.9, 25.7,
                r0+dr[0], r0+dr[1], {}, '90prime'),
            (#'apzpts/survey-ccds-mosaic-legacypipe.fits.gz',
                'apzpts/survey-ccds-mosaic.fits.gz',
                'survey-ccds-mosaic-psfzpts.fits',
                #'mosaic-psfzpts.fits',
                'z', 'MzLS z', 'z', 19.5, 25, 27, 0.262,
                #25.2, 26.8,
                z0+dz[0], z0+dz[1], bad_expid, 'mosaic'),
    ]:
        run(*X)    


