"""
Generally useful functions for other modules or repos
"""
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import fitsio
from glob import glob

try:
    from astrometry.util.fits import fits_table, merge_tables
except ImportError:
    pass

def inJupyter():
    return 'inline' in matplotlib.get_backend()
    
def save_png(outdir,fig_id):
    path= os.path.join(outdir,fig_id + ".png")
    if not os.path.isdir(outdir):
        os.makedirs(dirname)
    print("Saving figure", path)
    plt.tight_layout()
    plt.savefig(path, format='png', dpi=150)
    #plt.savefig(path, format='png',box_extra_artists=[xlab,ylab],
    #            bbox_inches='tight',dpi=150)
    if not inJupyter():
        plt.close()

def dobash(cmd):
  print('UNIX cmd: %s' % cmd)
  if os.system(cmd): raise ValueError

def fits2pandas(tab,attrs=None):
    """converts a fits_table into a pandas DataFrame

    Args:
      tab: fits_table()
      attrs: attributes or column names want in the DF
    """
    d={}
    if attrs is None:
        attrs= tab.get_columns()
    for col in attrs:
        d[col]= tab.get(col)
    df= pd.DataFrame(d)
    # Fix byte ordering from fits
    # https://stackoverflow.com/questions/18599579/pulling-multiple-non-consecutive-index-values-from-a-pandas-dataframe
    df= df.apply(lambda x: x.values.byteswap().newbyteorder())
    return df

def merge_tables_fns(self,fn_list,textfile=True,
                     shuffle=None):
    """concatenates fits tables
    shuffle: set to an integer to randomly reads up to the 
      first "shuffle" cats only
    """
    if shuffle:
        assert( isinstance(shuffle, int))
    if textfile: 
        fns=read_lines(fn_list)
    else:
        fns= fn_list
    if len(fns) < 1: raise ValueError('Error: fns=',fns)
    if shuffle:
        print('shuffling %d' % shuffle)
        seed=7
        np.random.seed(seed)
        inds= np.arange(len(fns)) 
        np.random.shuffle(inds) 
        fns= fns[inds]
        cats= []
    for i,fn in enumerate(fns):
        print('reading %s %d/%d' % (fn,i+1,len(fns)))
        if shuffle and i >= shuffle: 
            print('shuffle_1000 turned ON, stopping read') 
            break 
        try:
            tab= fits_table(fn) 
            cats.append( tab )
        except IOError:
            print('Fits file does not exist: %s' % fn)
    return merge_tables(cats, columns='fillzero')
