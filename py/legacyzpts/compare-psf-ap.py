from __future__ import print_function
import numpy as np
import pylab as plt
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence

plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.15, bottom=0.1, top=0.95, right=0.95)

def run(apfn,psffn,plotfn,tt,band,zplolo,zplo,zphi,pixscale,
        zpt_cut_lo, zpt_cut_hi, bad_expid):

    ps = PlotSequence('zp-' + plotfn)
    #A = fits_table('~desiproc/dr6/survey-ccds-mosaic-legacypipe.fits.gz')
    #A = fits_table('apzpts/survey-ccds-mosaic-legacypipe.fits.gz')
    #P = fits_table('survey-ccds-mosaic-psfzpts.fits')
    A = fits_table(apfn)
    P = fits_table(psffn)
    print(len(A), 'aperture')
    print(len(P), 'PSF')

    A.ccdzpt = A.zpt
    A.ccdphrms = A.phrms

    # P.ccdzpt = P.zpt
    # P.ccdphrms = P.phrms
    # P.ccdnmatch = P.nmatch_photom

    #A.about()
    #P.about()

    print('Aperture unique bands:', np.unique(A.filter))
    print('PSF unique bands:', np.unique(P.filter))

    A.cut(np.array([f.strip() == band for f in A.filter]))
    P.cut(np.array([f.strip() == band for f in P.filter]))
    print('Cut to', len(A), 'aperture in band', band)
    print('Cut to', len(P), 'PSF in band', band)

    #print('PSF error messages:')
    #print(np.unique(P.err_message))
    
    amap = dict([((expnum,ccdname.strip()),i) for i,(expnum,ccdname)
                 in enumerate(zip(A.expnum, A.ccdname))])

    pa = np.array([amap.get((expnum,ccdname.strip()), -1) for expnum,ccdname
                   in zip(P.expnum, P.ccdname)])

    print(np.sum(pa >= 0), 'match')

    P.cut([pa >= 0])
    A.cut(pa[pa >= 0])

    plt.clf()
    mn,mx = zplolo,zphi
    plt.plot(np.clip(A.ccdzpt, mn,mx), np.clip(P.ccdzpt,mn,mx), 'b.', alpha=0.1)
    #I = np.flatnonzero(A.ccd_cuts > 0)
    #plt.plot(np.clip(A.ccdzpt[I], mn,mx), np.clip(P[I].ccdzpt,mn,mx), 'r.', alpha=0.5)
    plt.xlabel('Aperture zeropoint')
    plt.ylabel('PSF zeropoint')
    #ax = plt.axis()
    #ax = [19.5,27,19.5,27]
    #mn,mx = min(ax[0], ax[2]), max(ax[1],ax[3])
    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.5)
    plt.axis([mn,mx,mn,mx])
    #plt.axis(ax)
    ps.savefig()

    plt.clf()
    plt.hist(P.fwhm, bins=50)
    plt.xlabel('PSF FWHM (pix)')
    ps.savefig()
    
    seeing = np.isfinite(P.fwhm) * P.fwhm * pixscale
    P.ccdzpt[np.logical_not(np.isfinite(P.ccdzpt))] = 0.
    I = np.flatnonzero(
        #(A.ccd_cuts == 0) *
        np.isfinite(A.ccdzpt) *
        (P.ccdzpt >= zpt_cut_lo) *
        (P.ccdzpt <= zpt_cut_hi) *
        (P.ccdphrms < 0.1) *
        (P.ccdrarms  < 0.25) *
        (P.ccddecrms < 0.25) *
        (P.exptime > 30) *
        (np.abs(P.ccdzpt - P.zpt) < 0.25) *
        (seeing < 3.0) * (seeing > 0)
    )
    #(seeing < 2.5) * (seeing > 0)

    A.cut(I)
    P.cut(I)
    print(len(A), 'pass cuts')

    #I = np.flatnonzero(np.logical_not(np.isfinite(A.ccdzpt)))
    #print(len(I), 'AP zpts are NaN')
    #A.ccdzpt[I] = 0.

    print('After cuts: AP  zpt range', A.ccdzpt.min(), A.ccdzpt.max())
    print('After cuts: PSF zpt range', P.ccdzpt.min(), P.ccdzpt.max())

    #print('PSF error messages:')
    #print(np.unique(P.err_message))

    bad = np.array([expnum in bad_expid for expnum in P.expnum])
    print(sum(bad), 'exposures are in the bad_expid file')
    
    plt.clf()
    mn,mx = zplo,zphi
    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.5)
    plt.plot(np.clip(A.ccdzpt, mn,mx), np.clip(P.ccdzpt,mn,mx), 'b.', alpha=0.25)
    plt.xlabel('Aperture zeropoint')
    plt.ylabel('PSF zeropoint')
    plt.axis([mn,mx,mn,mx])
    plt.title(tt)
    ps.savefig()
    
    plt.clf()
    plt.axhline(0, color='k', alpha=0.5)
    plt.plot(np.clip(A.ccdzpt, mn,mx), np.clip(P.ccdzpt - A.ccdzpt, -1, 1),
             'b.', alpha=0.25)
    plt.xlabel('Aperture zeropoint')
    plt.ylabel('PSF - Aperture zeropoint')
    plt.axis([mn,mx, -1,1])
    plt.title(tt)
    ps.savefig()
    
    # plt.clf()
    # plt.axhline(0, color='k', alpha=0.5)
    # plt.scatter(np.clip(A.ccdzpt, mn,mx), np.clip(P.ccdzpt - A.ccdzpt, -1, 1),
    #             c=P.ccdphrms)
    # plt.colorbar()
    # plt.xlabel('Aperture zeropoint')
    # plt.ylabel('PSF - Aperture zeropoint')
    # plt.axis([mn,mx, -1,1])
    # plt.title('color: PSF ccdphrms')
    # ps.savefig()
    
    plt.clf()
    mx = 0.1
    plt.hist(np.clip(A.ccdphrms, 0, mx), bins=50, range=(0,mx), histtype='step', color='r', label='Aperture')
    plt.hist(np.clip(P.ccdphrms, 0, mx), bins=50, range=(0,mx), histtype='step', color='b', label='PSF')
    plt.xlim(0,mx)
    plt.xlabel('CCD phrms')
    plt.legend()
    ps.savefig()
    
    plt.clf()
    plt.subplot(2,1,1)
    mn,mx = -0.25, 0.25
    plt.hist(np.clip(P.ccdzpt - A.ccdzpt, mn, mx), bins=50, range=(mn, mx),
             histtype='step')
    plt.xlabel('PSF - Aperture ccdzpt')
    plt.xlim(mn, mx)
    plt.subplot(2,1,2)
    plt.hist(np.clip(P.ccdzpt - A.ccdzpt, mn, mx), bins=50, range=(mn, mx),
             histtype='step', log=True)
    plt.xlabel('PSF - Aperture ccdzpt')
    plt.xlim(mn, mx)
    plt.suptitle(tt)
    ps.savefig()


    plt.clf()
    mn,mx = 0,0.25
    plt.hist(np.clip(P.ccdrarms ,mn,mx), bins=50, range=(mn,mx), histtype='step', color='b', label='RA' , log=True)
    plt.hist(np.clip(P.ccddecrms,mn,mx), bins=50, range=(mn,mx), histtype='step', color='g', label='Dec', log=True)
    plt.xlim(mn,mx)
    plt.legend()
    plt.xlabel('Astrometric scatter')
    plt.suptitle(tt)
    ps.savefig()

    plt.clf()
    plt.hist(P.ccdskycounts,  bins=50, log=True)
    plt.xlabel('CCD sky counts')
    plt.suptitle(tt)
    ps.savefig()

    plt.clf()
    plt.hist(P.exptime,  bins=50, log=True)
    plt.xlabel('Exptime (s)')
    plt.suptitle(tt)
    ps.savefig()

    plt.clf()
    plt.hist(P.ccdzpt - P.zpt,  bins=50, log=True)
    plt.xlabel('CCD zpt - average zpt')
    plt.suptitle(tt)
    ps.savefig()

    plt.clf()
    plt.hist(P.fwhm * pixscale,  bins=50, log=True)
    plt.xlabel('Seeing (arcsec)')
    plt.suptitle(tt)
    ps.savefig()
    
    J = np.flatnonzero(np.logical_or((P.ccdzpt - A.ccdzpt) < -0.1,
                                     (P.ccdzpt - A.ccdzpt) >  0.2))
    print('Some bad ones:')
    for j in J:
        print(P.image_filename[j].strip(), 'PSF nmatch', P.ccdnmatch[j], 'phrms', P.ccdphrms[j], 'AP phrms', A.ccdphrms[j], 'exptime', A.exptime[j], 'seeing', A.fwhm[j] * pixscale)
        print('  AP err', A.err_message[j], 'CCD cuts', A.ccd_cuts[j])
        print('  Expnum', P.expnum[j], 'Bad expid:', bad_expid.get(int(P.expnum[j]), '(none)'))

        ttxt = '%s %s %i' % (P.image_filename[j].strip().replace('.fits.fz','').replace('_ooi','').replace('_zd',''),
                          P.ccdname[j].strip(), P.expnum[j])

        plt.clf()
        #for i in range(4):
        #plt.subplot(2,2,i+1)
        img = fitsio.read(P.image_filename[j].strip(), ext=P.ccdname[j].strip()) #ext=i+1)
        mn,mx = np.percentile(img.ravel(), [25,98])
        plt.imshow(img, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.title(ttxt)
        ps.savefig()

        H,W = img.shape
        #plt.clf()
        #plt.imshow(img[H//2-250:H//2+250, W//2-250:W//2+250],
        #               interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.axis([W//2-250, W//2+250, H//2-250, H//2+250])
        plt.title(ttxt)
        ps.savefig()

        
        plt.clf()
        plt.subplot(1,2,1)
        fn = P.image_filename[j].strip().replace('_ooi_','_ood_')
        dq = fitsio.read(fn, ext=P.ccdname[j].strip())
        mn,mx = 0, 1
        plt.imshow(dq, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.xticks([]); plt.yticks([])
        plt.title('DQ')
        
        plt.subplot(1,2,2)
        fn = P.image_filename[j].strip().replace('_ooi_','_oow_')
        wt = fitsio.read(fn, ext=P.ccdname[j].strip())
        mn,mx = 0, wt.max()
        plt.imshow(wt, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.xticks([]); plt.yticks([])
        plt.title('WT')
        plt.suptitle(ttxt)
        ps.savefig()


if __name__ == '__main__':

    bad_expid = {}
    f = open('bad_expid.txt')
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

    for X in [
            (#'apzpts/survey-ccds-90prime-legacypipe.fits.gz',
                'apzpts/survey-ccds-90prime.fits.gz',
                'survey-ccds-90prime-psfzpts.fits',
                #'90prime-psfzpts.fits',
                'g', 'BASS g', 'g', 20, 25, 26.25, 0.45,
                25.2, 26.0, {}),
            (#'apzpts/survey-ccds-90prime-legacypipe.fits.gz',
                'apzpts/survey-ccds-90prime.fits.gz',
                'survey-ccds-90prime-psfzpts.fits',
                #'90prime-psfzpts.fits',
                'r', 'BASS r', 'r', 19.5, 24.75, 25.75, 0.45,
                24.9, 25.7, {}),
            (#'apzpts/survey-ccds-mosaic-legacypipe.fits.gz',
                'apzpts/survey-ccds-mosaic.fits.gz',
                'survey-ccds-mosaic-psfzpts.fits',
                #'mosaic-psfzpts.fits',
                'z', 'MzLS z', 'z', 19.5, 25, 27, 0.262,
                25.2, 26.8, bad_expid),
    ]:
        run(*X)    

