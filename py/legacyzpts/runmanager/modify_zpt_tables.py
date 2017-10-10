import numpy as np

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec

def add_zpt_to_legacypipe(zpt_fn,leg_fn, new_leg_fn):
  """Adds columns from zpt table to legacypipe table
  """
  zpt=fits_table(zpt_fn)
  leg=fits_table(leg_fn)
  assert(len(zpt) == len(leg))
  m1, m2, _ = match_radec(zpt.ra, zpt.dec, leg.ra,leg.dec,1./3600.0,nearest=True)
  assert(len(m1) == len(zpt))
  zpt= zpt[m1]
  leg= leg[m2]
  assert(np.all(zpt.ccdname[:1000] == leg.ccdname[:1000]))
  assert(np.all(zpt.expnum[:1000] == leg.expnum[:1000]))

  leg.set('phrms',zpt.get('phrms'))
  leg.writeto(new_leg_fn)
  print('Wrote %s' % new_leg_fn)
