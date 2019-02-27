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

MJD_EARLY_DECAM = 56730.

def detrend_decam_zeropoints(P):
    '''
    Per Arjun's email 2019-02-27 "Zeropoint variations with MJD for
    DECam data", correct zeropoints for trends with airmass and MJD
    before making too-big/too-small cuts.
    '''
    zpt_corr = P.ccdzpt.copy()
    ntot = 0
    for band,k in [('g', 0.173),
                   ('r', 0.090),
                   ('z', 0.060),]:
        I = np.flatnonzero((P.band == band) * (P.airmass >= 1.0))
        if len(I) == 0:
            continue
        ntot += len(I)
        zpt_corr[I] -= k * (P.airmass[I] - 1.0)

    if ntot < len(P):
        print('In detrend_decam_zeropoints: did not detrend for airmass variation for', len(P)-ntot, 'CCDs due to unknown band or bad airmass')

    mjd_terms = [
        ('g', 25.08, [
            (   0.0,  160.0, 25.170, 25.130, 25.170,  -2.5001e-04),
            ( 160.0,  480.0, 25.180, 25.080, 25.230,  -3.1250e-04),
            ( 480.0,  810.0, 25.080, 25.080, 25.080,   0.0000e+00),
            ( 810.0,  950.0, 25.130, 25.130, 25.130,   0.0000e+00),
            ( 950.0, 1250.0, 25.130, 25.040, 25.415,  -2.9999e-04),
            (1250.0, 1650.0, 25.080, 25.000, 25.330,  -2.0000e-04),
            (1650.0, 1900.0, 25.270, 25.210, 25.666,  -2.4001e-04),]),
        ('r', 25.29, [
            (   0.0,  160.0, 25.340, 25.340, 25.340,   0.0000e+00),
            ( 160.0,  480.0, 25.370, 25.300, 25.405,  -2.1876e-04),
            ( 480.0,  810.0, 25.300, 25.280, 25.329,  -6.0602e-05),
            ( 810.0,  950.0, 25.350, 25.350, 25.350,   0.0000e+00),
            ( 950.0, 1250.0, 25.350, 25.260, 25.635,  -3.0000e-04),
            (1250.0, 1650.0, 25.320, 25.240, 25.570,  -2.0000e-04),
            (1650.0, 1900.0, 25.440, 25.380, 25.836,  -2.4001e-04),]),
        ('z', 24.92, [
            (   0.0,  160.0, 24.970, 24.970, 24.970,   0.0000e+00),
            ( 160.0,  480.0, 25.030, 24.950, 25.070,  -2.5000e-04),
            ( 480.0,  760.0, 24.970, 24.900, 25.090,  -2.5000e-04),
            ( 760.0,  950.0, 24.900, 25.030, 24.380,   6.8422e-04),
            ( 950.0, 1150.0, 25.030, 24.880, 25.743,  -7.5001e-04),
            (1150.0, 1270.0, 24.880, 25.030, 23.442,   1.2500e-03),
            (1270.0, 1650.0, 25.030, 24.890, 25.498,  -3.6842e-04),
            (1650.0, 1900.0, 25.070, 24.940, 25.928,  -5.2000e-04),]),]

    ntot = 0
    mjd0 = 56658.5
    for band,zpt0,terms in mjd_terms:
        I = np.flatnonzero((P.band == band) * (P.mjd_obs > 0))
        if len(I) == 0:
            continue
        day = P.mjd_obs[I] - mjd0
        # Piecewise linear function
        for day_i, day_f, zpt_i, zpt_f, c0, c1 in terms:
            c1 = (zpt_f - zpt_i) / (day_f - day_i)
            Jday = (day >= day_i) * (day < day_f)
            J = I[Jday]
            if len(J) == 0:
                continue
            ntot += len(J)
            zpt_corr[J] += zpt0 - (c0 + c1*day[Jday])
    if ntot < len(P):
        print('In detrend_decam_zeropoints: did not detrend for temporal variation for', len(P)-ntot, 'CCDs due to unknown band or MJD_OBS')

    return zpt_corr

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

    if camera == 'decam':
        ccdzpt = detrend_decam_zeropoints(P)
    else:
        ccdzpt = P.ccdzpt

    cuts = [
        ('not_grz',   np.array([f.strip() not in keys for f in P.filter])),
        ('zpt_small', np.array([ccdzpt < zpt_cut_lo.get(f,0) for f,ccdzpt in zip(P.filter, ccdzpt)])),
        ('zpt_large', np.array([ccdzpt > zpt_cut_hi.get(f,0) for f,ccdzpt in zip(P.filter, ccdzpt)])),
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
    import sys
    # DECam updated for DR8, post detrend_decam_zeropoints.

    camera = 'decam'
    bad_expid = read_bad_expid('obstatus/bad_expid.txt')

    g0 = 25.08
    r0 = 25.29
    i0 = 25.26
    z0 = 24.92
    dg = (-0.5, 0.25)
    di = (-0.5, 0.25)
    dr = (-0.5, 0.25)
    dz = (-0.5, 0.25)
    zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], i=i0+dr[0], z=z0+dz[0])
    zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], i=i0+dr[1], z=z0+dz[1])

    TT = []
    for band in ['g','r','z']:
        infn = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8/DECaLS/survey-ccds-decam-%s.fits.gz' % band

        T = fits_table(infn)
        print('Read', len(T), 'CCDs for', band)
        print('Initial:', np.sum(T.ccd_cuts == 0), 'CCDs pass cuts')

        plt.clf()
        detrend = detrend_decam_zeropoints(T)
        plt.subplot(2,1,1)
        plt.plot(T.mjd_obs, T.ccdzpt, 'b.')
        plt.subplot(2,1,2)
        plt.plot(T.mjd_obs, detrend, 'b.')
        plt.savefig('detrend-%s.png' % band)
        
        psf_zeropoint_cuts(T, 0.262, zpt_lo, zpt_hi, bad_expid, camera)
        print('Final:', np.sum(T.ccd_cuts == 0), 'CCDs pass cuts')
        TT.append(T)

    T = merge_tables(TT)
    fn = 'survey-ccds.fits'
    T.writeto(fn)
    from legacypipe.create_kdtrees import create_kdtree
    kdfn = 'survey-ccds.kd.fits'
    create_kdtree(fn, kdfn, True)
    print('Wrote', kdfn)

    sys.exit(0)

    ################################

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


