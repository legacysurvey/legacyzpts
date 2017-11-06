import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import numbers
import seaborn as sns

import fitsio

from legacyzpts.legacy_zeroints import create_legacypipe_table

CAMERAS= ['decam','90prime','mosaic']

class LegacypipeCuts(object):
	"""Applies legacyipe cuts

	Args: 
		T_leg: '*-zpt.fits' table
	"""
	def __init__(self,T_zpt,camera=None):
		assert(camera in CAMERAS)
	    self.camera= camera
		self.ccds= self.create_legacypipe_table(T=T_zpt.copy(), camera=self.camera)
		self.good= np.ones(len(self.T_leg),bool)

    def photometric(self):
        z0 = self.nominal_zeropoints()
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
             (ccds.zpt > self.max_zeropoint(z0))),
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
   
	def extra(self):
		if self.camera == 'mosaic':
			for name,crit in [
				('sky too bright', 
				 (self.ccds.ccdskycounts >= 150)),
				('only z band',
				 self.ccds.filter != 'z'),
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
            

class LegacyzptsCuts(object):
	"""Applies legacyzpts cuts

	Args:
		T_zpt: '*-zpts.fits' table
	"""
	
	def __init__(self,T_zpt,camera=None):
		assert(camera in CAMERAS)
		self.camera= camera
		self.T= T_zpt.copy()
		self.good= np.ones(len(T),bool)
		self.df= pd.DataFrame({'err':big2small_endian(T.err_message)})
        self.df['err']= self.df['err'].str.strip()

	def err_message(self):
        self.good[self.df['err'].str.len() > 0]= False

	def third_pix(self):
        # The 1/3-pixel shift problem was fixed in hardware on MJD 57674,
        # so only check for problems in data before then.
        self.good[((self.T.has_yshift == False) & 
				   (self.T.mjd_obs < 57674.))]= False


if __name__ == '__main__':
	camera='decam'
	T_zpt= fits_table(zpt_fn)
	zpt_cuts= LegacypipeCuts(T_zpt,camera)
	zpt_cuts.err_message()
	zpt_cuts.third_pix()

	leg_cuts= LegacypipeCuts(T_zpt,camera)
	leg_cuts.photometric()
	leg_cuts.bad_exposures()
	leg_cuts.extra()

	T_zpt.set('good',((zpt_cuts.good) & 
					  (leg_cuts.good)))
	fn=zpt_fn.replace('.fits.fz','').replace('.fits','') + 'survey-ccds-'+camera+'.fits'
	T_zpt.writeto(fn)
	print('Wrote %s' % fn)
	T_leg= create_legacypipe_table(T_zpt, camera=camera)
	T_leg.set('good',T_zpt.good)
	fn=fn.replace('.fits','-legacypipe.fits')
	T_zpt.writeto(fn)
	print('Wrote %s' % fn)
	
	

	
	
