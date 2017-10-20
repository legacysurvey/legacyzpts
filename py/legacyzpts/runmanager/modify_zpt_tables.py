import numpy as np

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec

CAMERAS=['decam','mosaic','90prime']

def add_to_legacypipe(zpt_fn,leg_fn, new_leg_fn):
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

def cleanup_zpt_table(fn,camera):
  """Removes unneeded cols
  
  Args:
    fn: filename for '*-zpt.fits' table
  """
  assert(camera in CAMERAS)
  t=fits_table(fn)
  del_cols=['mdncol','nstarfind','skycounts_a',
            'skyrms_a','skyrms_b','skyrms_c','skyrms_d',
            'skyrms_clip','skyrms_clip_sm','skyrms_sigma',
            'skyrms_sm']
  for col in del_cols:
    if col in t.get_columns():
      t.delete_column(col)
  if camera == '90prime':
    t.set('gain',np.zeros(len(t)) + 1.4)
  fnout= fn.replace('.fits','').replace('.gz','')
  fnout+= '_cleaned.fits'
  t.writeto(fnout)
  print('wrote %s' % fnout)
 

def cleanup_star_table(fn,which='photom'):
  """Removes unneeded cols from wither astrom or photom stars table
  
  Args:
    fn: filename for 'stars-photom' or '-astrom' table
    which: astrom,photom
  """
  assert(which in ['photom','astrom'])
  t=fits_table(fn)
  del_cols={
    "both":['image_hdu','amplifier','ps1_gicolor',
            'radiff_ps1','decdiff_ps1','daofind_x','daofind_y',
            'mycuts_x','mycuts_y'],
    "photom":['radiff','decdiff','gaia_ra','gaia_dec','gaia_g'],
    "astrom":['dmagall','apmag','apflux','apskyflux','apskyflux_perpix',
              'ps1_mag','ps1_g','ps1_r','ps1_i','ps1_z','gaia_g',
              'ra','dec','x','y','gaia_ra','gaia_dec']
           }
  for col in del_cols['both']:
    if col in t.get_columns():
      t.delete_column(col)
  for col in del_cols[which]:
    if col in t.get_columns():
      t.delete_column(col)
  fnout= fn.replace('.fits','').replace('.gz','')
  fnout+= '_cleaned.fits'
  t.writeto(fnout)
  print('wrote %s' % fnout)
 
