from __future__ import print_function
import numpy as np
import pylab as plt
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence
from collections import Counter

plt.figure(figsize=(10,7))
plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.15, bottom=0.1, top=0.95, right=0.95)

def run(apfn,psffn,plotfn,tt,band,zplolo,zplo,zphi,pixscale,
        zpt_cut_lo, zpt_cut_hi, bad_expid):

    ps = PlotSequence('zp-' + plotfn)
    A = fits_table(apfn)
    P = fits_table(psffn)
    print(len(A), 'aperture')
    print(len(P), 'PSF')

    A.ccdzpt = A.zpt
    A.ccdphrms = A.phrms

    # P.ccdzpt = P.zpt
    # P.ccdphrms = P.phrms
    # P.ccdnmatch = P.nmatch_photom

    print('Aperture unique bands:', np.unique(A.filter))
    print('PSF unique bands:', np.unique(P.filter))

    A.cut(np.array([f.strip() == band for f in A.filter]))
    P.cut(np.array([f.strip() == band for f in P.filter]))
    print('Cut to', len(A), 'aperture in band', band)
    print('Cut to', len(P), 'PSF in band', band)

    ## PSF zeropoints cuts

    seeing = np.isfinite(P.fwhm) * P.fwhm * pixscale
    P.ccdzpt[np.logical_not(np.isfinite(P.ccdzpt))] = 0.
    I = np.flatnonzero(
        #(A.ccd_cuts == 0) *
        #np.isfinite(A.ccdzpt) *
        (P.ccdzpt >= zpt_cut_lo) *
        (P.ccdzpt <= zpt_cut_hi) *
        (P.ccdphrms < 0.1) *
        (P.ccdrarms  < 0.25) *
        (P.ccddecrms < 0.25) *
        (P.exptime > 30) *
        (np.abs(P.ccdzpt - P.zpt) < 0.25) *
        (seeing < 3.0) * (seeing > 0)
    )
    P.cut(I)
    print(len(P), 'pass cuts')

    bad = np.array([expnum in bad_expid for expnum in P.expnum])
    print(sum(bad), 'CCDs are in the bad_expid file')

    P.cut(np.logical_not(bad))
    print(len(P), 'pass not-in-bad-expid cut')

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

    plt.clf()
    plt.hist(P.skyrms,  bins=50, log=True)
    plt.xlabel('Skyrms')
    plt.suptitle(tt)
    ps.savefig()

    plt.clf()
    plt.hist(P.sig1,  bins=50, log=True)
    plt.xlabel('sig1')
    plt.suptitle(tt)
    ps.savefig()

    if False:
        ### Sort by expnum and CCDname
        I = np.lexsort((P.ccdname, P.expnum))
        P.cut(I)

        bad = np.array([expnum in bad_expid for expnum in P.expnum])
        print(sum(bad), 'CCDs are in the bad_expid file')

        expnums = np.unique(P.expnum)
        print(len(expnums), 'unique exposure numbers')
        ebad = np.array([expnum in bad_expid for expnum in expnums])
        print(sum(ebad), 'exposures are in the bad_expid file')
    
        J = np.flatnonzero(bad)
    
        badtxt = [bad_expid[expnum] for expnum in P.expnum[J]]
    
        known_bad = np.zeros(len(J), bool)
        known_ok  = np.zeros(len(J), bool)
    
        # These are "not necessarily bad": low transparency, and bad amps
        # (not processed by the CP in the first place; ie, only 3/4 CCDs exist)
        for word in ['bad amp', 'trans', 'vignet', 'depth factor', 'depfac', 'depth',
                     'expfactor', 'sky background',
                     'pass2 image in poor', 'pass2 seeing', 'pass1 image taken',
                     'pointing', 'clouds']:
            # Find indices in J where 'word' is found in the bad_expid entry.
            I = np.array([i for i in range(len(J)) if word in badtxt[i].lower()])
            print(len(I), 'with "%s" in description' % word)
            known_ok[I] = True
    
        for word in ['4maps', 'focus', 'jump', 'tails', 'trail', 'tracking',
                     'elong', 'triangle']:
            # Find indices in J where 'word' is found in the bad_expid entry.
            I = np.array([i for i in range(len(J)) if word in badtxt[i].lower()])
            print(len(I), 'with "%s" in description' % word)
            known_bad[I] = True
    
        known = np.logical_or(known_bad, known_ok)
        Jun = J[known == False]
        print(len(Jun), 'unclassified bad exposures')
        ub = Counter([badtxt[i] for i in np.flatnonzero(known == False)])
        print('Unclassified descriptions:')
        for k,n in ub.most_common():
            print(n, k)
    
        isbad = np.ones(len(J), bool)
        isbad[known_ok] = False
        isbad[known_bad] = True
        J = J[isbad]
        print('Keeping', len(J), 'of the bad-expid CCDs')
    
        keep = np.ones(len(P), bool)
        keep[J] = False
        print('Keeping', np.sum(keep), 'of', len(keep), 'CCDs')
        P.cut(keep)





    amap = dict([((expnum,ccdname.strip()),i) for i,(expnum,ccdname)
                 in enumerate(zip(A.expnum, A.ccdname))])

    pa = np.array([amap.get((expnum,ccdname.strip()), -1) for expnum,ccdname
                   in zip(P.expnum, P.ccdname)])

    print(np.sum(pa >= 0), 'match')

    P.cut([pa >= 0])
    A.cut(pa[pa >= 0])

    I = np.isfinite(A.ccdzpt)
    P.cut(I)
    A.cut(I)
    print('Cut to', len(P), 'with finite aperture zeropoints')

    print('AP  zpt range', A.ccdzpt.min(), A.ccdzpt.max())
    print('PSF zpt range', P.ccdzpt.min(), P.ccdzpt.max())

    plt.clf()
    mn,mx = zplolo,zphi
    plt.plot(np.clip(A.ccdzpt, mn,mx), np.clip(P.ccdzpt,mn,mx), 'b.', alpha=0.1)
    plt.xlabel('Aperture zeropoint')
    plt.ylabel('PSF zeropoint')
    plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.5)
    plt.axis([mn,mx,mn,mx])
    ps.savefig()


    #print('Sampling of bad_expids:')
    #for j in J[np.random.permutation(len(J))[:20]]:
    # for j in Jun:
    #     print(P.image_filename[j].strip(), 'PSF nmatch', P.ccdnmatch[j], 'phrms', P.ccdphrms[j], 'AP phrms', A.ccdphrms[j], 'exptime', A.exptime[j], 'seeing', A.fwhm[j] * pixscale)
    #     print('  AP err', A.err_message[j], 'CCD cuts', A.ccd_cuts[j])
    #     print('  Expnum', P.expnum[j], 'Bad expid:', bad_expid.get(int(P.expnum[j]), '(none)'))
    # 
    #     ttxt = '%s %s %i %s' % (P.image_filename[j].strip().replace('.fits.fz','').replace('_ooi','').replace('_zd',''),
    #                              P.ccdname[j].strip(), P.expnum[j], bad_expid[P.expnum[j]])
    # 
    #     # plt.clf()
    #     # img = fitsio.read(P.image_filename[j].strip(), ext=P.ccdname[j].strip())
    #     # mn,mx = np.percentile(img.ravel(), [25,98])
    #     # plt.imshow(img, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
    #     # plt.title(ttxt)
    #     # ps.savefig()
    #     # H,W = img.shape
    #     # plt.axis([W//2-250, W//2+250, H//2-250, H//2+250])
    #     # plt.title(ttxt)
    #     # ps.savefig()
    # 
    #     plt.clf()
    # 
    #     plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    #     img = fitsio.read(P.image_filename[j].strip(), ext=P.ccdname[j].strip())
    #     H,W = img.shape
    # 
    #     fn = P.image_filename[j].strip().replace('_ooi_','_ood_')
    #     dq = fitsio.read(fn, ext=P.ccdname[j].strip())
    #     fn = P.image_filename[j].strip().replace('_ooi_','_oow_')
    #     wt = fitsio.read(fn, ext=P.ccdname[j].strip())
    #     wt[dq != 0] = 0.
    # 
    #     binned,nil = bin_image(img, wt, 4)
    #     mn,mx = np.percentile(binned.ravel(), [40,99])
    #     plt.imshow(binned, interpolation='nearest', origin='lower', vmin=mn, vmax=mx,
    #                cmap='gray', extent=[0,W,0,H])
    # 
    #     #mn,mx = np.percentile(img.ravel(), [40,99])
    #     #plt.imshow(img, interpolation='nearest', origin='lower', vmin=mn, vmax=mx,
    #     #           cmap='gray')
    # 
    #     plt.subplot2grid((2, 3), (0, 2))
    #     x0,x1,y0,y1 = W//2-200, W//2+200, H//2-200, H//2+200
    #     #mn = np.percentile(img.ravel(), 40)
    #     #mx = img[y0:y1, x0:x1].max()
    #     mn,mx = np.percentile(img.ravel(), [40,99.5])
    #     plt.imshow(img, interpolation='nearest', origin='lower', vmin=mn, vmax=mx,
    #                cmap='gray')
    #     plt.axis([x0,x1,y0,y1])
    # 
    #     plt.subplot2grid((2, 3), (1, 2))
    #     mn,mx = 0, np.percentile(wt, 95)
    #     plt.imshow(wt, interpolation='nearest', origin='lower', vmin=mn, vmax=mx, cmap='gray')
    #     plt.xticks([]); plt.yticks([])
    # 
    #     plt.suptitle(ttxt, fontsize=8)
    #     ps.savefig()



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

def bin_image(data, invvar, S):
    # rebin image data
    H,W = data.shape
    sH,sW = (H+S-1)//S, (W+S-1)//S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    newiv = np.zeros((sH,sW), dtype=invvar.dtype)
    for i in range(S):
        for j in range(S):
            iv = invvar[i::S, j::S]
            subh,subw = iv.shape
            newdata[:subh,:subw] += data[i::S, j::S] * iv
            newiv  [:subh,:subw] += iv
    newdata /= (newiv + (newiv == 0)*1.)
    newdata[newiv == 0] = 0.
    return newdata,newiv

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


