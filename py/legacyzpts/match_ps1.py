import numpy as np
import os
import matplotlib.pyplot as plt

import fitsio

if __name__ == '__main__':
  from astrometry.util.fits import fits_table, merge_tables
  from astrometry.util.util import wcs_pv2sip_hdr
  from astrometry.libkd.spherematch import match_radec
  from legacypipe.survey import LegacySurveyData, wcs_for_brick

from legacyanalysis.ps1cat import ps1cat, ps1_to_decam


from legacyzpt.qa.paper_plots import myscatter

PROJ_DIR= "/project/projectdirs/cosmo/work/legacysurvey/dr5/DR5_out"

def get_tractorfn(brick):
  return os.path.join(PROJ_DIR, 'tractor/%s/tractor-%s.fits' %
                      (brick[:3],brick))

def get_ps1_colorterm(ps1stars, band, camera='decam'):
  if camera == 'decam':
    return ps1_to_decam(ps1stars, band)
  else:
    raise ValueError('camera %s not supported' % camera)

def magAB_to_nanomaggie_flux(magAB):
  return 1E9*10**(-0.4*magAB)

def nanomaggie_flux_to_magAB(nanoflux):
  return -2.5*np.log10(nanoflux) + 2.5*9



brick="0657m042"
T= fits_table( get_tractorfn(brick) )
# z-band, type = PSF
band_type= (np.char.strip(T.type) == 'PSF')
T.cut(band_type)

survey = LegacySurveyData()
brickinfo = survey.get_brick_by_name(brick)
brickwcs = wcs_for_brick(brickinfo)
#wcs_pv2sip_hdr(self.hdr) # PV distortion
#hdulist= fitsio.FITS(self.fn)
#image_hdu= hdulist[ext].get_extnum() #NOT ccdnum in header!
#hdr = fitsio.read_header(fn, ext=ext)
 
ps1 = ps1cat(ccdwcs=brickwcs).get_stars() #magrange=(15, 22))
if len(ps1) == 0:
    raise ValueError("no PS1 ccds in region")

gicolor= ps1.median[:,0] - ps1.median[:,2]
good = ((ps1.nmag_ok[:, 0] > 0) & 
        (ps1.nmag_ok[:, 1] > 0) &
        (ps1.nmag_ok[:, 2] > 0) &
        (gicolor > 0.4) &
        (gicolor < 2.7))
ps1.cut(good)


# Match GAIA and Our Data
gdec= ps1.dec_ok-ps1.ddec/3600000.
gra= ps1.ra_ok-ps1.dra/3600000./np.cos(np.deg2rad(gdec))
m1, m2, d12 = match_radec(T.ra, T.dec, gra, gdec, 1./3600.0,\
                          nearest=True)

# Each source has grz
band='r'
colorterm = get_ps1_colorterm(ps1.median[m2, :], band)
ps1band = ps1cat.ps1band[band]
# g-band DECAM,MzLS,or BASS = g-band PS1 - poly(gicolor, gcoeff)
ps1_mag = ps1.median[m2, ps1band] + colorterm

trac_mag= nanomaggie_flux_to_magAB( T[m1].get('flux_%s' % band) )

#def plot_dmag_ps1_mag
fig,ax=plt.subplots()
myscatter(ax,ps1_mag, trac_mag-ps1_mag, 
          color='b',m='o',s=10.,alpha=0.75)
xlab= ax.set_xlabel('ps1 mag')
ylab= ax.set_ylabel('dmag (tractor - ps1)')
savefn= 'ps1_trac_%s.png' % band
plt.savefig(savefn, bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
print('Wrote %s' % savefn)
