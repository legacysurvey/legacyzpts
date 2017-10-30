import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from collections import defaultdict
from scipy.stats import sigmaclip
import pandas as pd
import numbers
import seaborn as sns

import fitsio
try:
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.libkd.spherematch import match_radec
except ImportError:
    pass            

from legacyzpts.qa import params 
from legacyzpts.qa.params import band2color,col2plotname

CAMERAS = ['90prime','mosaic','decam']

def big2small_endian(fits_table_array):
    """fits is big endian, pandas is small

    https://stackoverflow.com/questions/30283836/creating-pandas-dataframe-from-numpy-array-leads-to-strange-errors
    """
    return np.array(fits_table_array).byteswap().newbyteorder()


class QaPlots(object):
    """do QA on a -zpt.fits file

    Example:
        qa= QaPlots('decam','path/to/c4d-zpt.fits')
        qa.print_errs_and_nans()
        qa.ccds_per_exposure()
        qa.df_plots()
    """

    def __init__(self,camera,zpt_table_fn):
        assert(camera in CAMERAS)
        self.camera= camera
        self.zpt= fits_table(zpt_table_fn)
        self.add_fields()
        self.df= pd.DataFrame({key:self.zpt.get(key)
                               for key in ['filter', #'expnum',
                               'raoff','decoff',
                               'zpt','phrms','radecrms','err_message']})
        self.df['expnum']= big2small_endian(self.zpt.expnum)
        self.df['errs']= self.df['err_message'].str.strip()
        self.df['good']= self.df['errs'].str.len() == 0

    def print_errs_and_nans(self):
        errs= list(set(self.df['errs']))
        print('Error Messages= ',errs)
        for err in errs:
        print('%d/%d: %s' % 
              (self.df[self.df['errs'] == err].shape[0],
               self.df.shape[0],
               err))
        for col in self.zpt.get_columns():
            if isinstance(self.zpt.get(col)[0], numbers.Real):
                hasNan= np.isfinite(self.zpt.get(col)) == False
                if len(self.zpt.get(col)[(self.df['good']) & (hasNan)]) > 0:
                    print("%s: %d Nans" % (col,len(self.zpt.get(col)[(self.df['good']) & (hasNan)])))
 
    def ccds_per_exposure(self):
        counts= self.df['expnum'].value_counts().rename('count').to_frame()
        minccds={"decam":60,"mosaic":4,"90prime":4}
        ltMin= counts['count'] < minccds[self.camera]
        if counts[ltMin].shape[0] > 0:
            print("These have too FEW ccds")
            print(counts[ltMin])
        else:
            print("Every exposure has the right number of ccds")

    def vline(self,x,**kwargs):
        plt.axvline(0, **kwargs)
    def hline(self,x,**kwargs):
        plt.axhline(0, **kwargs)

    def df_plots(self):
        # https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6
        # Scatter
        for x,y in [('raoff','decoff'),('phrms','radecrms')]:
            g = sns.FacetGrid(self.df[self.df['good']], col="filter") 
            g.map(plt.scatter, x,y,s=10,alpha=0.4)
            g.map(self.vline, x,c='k',ls='--')
            g.map(self.hline, y,c='k',ls='--')
            plotname= '%s_%s_%s.png' % (self.camera,x,y)
            g.savefig(plotname)
            print('Wrote %s' % plotname)
        # Hist
        for x in ['zpt']:
            g = sns.FacetGrid(self.df[self.df['good']], col="filter") 
            g.map(sns.distplot, x)
            plotname= '%s_%s.png' % (self.camera,x)
            g.savefig(plotname)
            print('Wrote %s' % plotname)
            xlim= {'decam':(25.5, 27.2),
                   'mosaic':(25.5, 27.2),
                   '90prime':(24.5, 26.5)}
            g.set(xlim=xlim[self.camera])
            plotname= '%s_%s_zoom.png' % (self.camera,x)
            g.savefig(plotname)
            print('Wrote %s' % plotname)

  
    def add_fields(self):
        self.zpt.set('radecrms',np.sqrt(self.zpt.rarms**2 + self.zpt.decrms**2))
  

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--camera',choices=['decam','90prime','mosaic'],required=True)
    parser.add_argument('--zptfn',help='a *-zpt.fits fileanme',required=True)
    args = parser.parse_args()

    qa= QaPlots(args.camera,args.zptfn)
    qa.print_errs_and_nans()
    qa.ccds_per_exposure()
    qa.df_plots()
