"""
Generally useful functions for other modules or repos
"""
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import os
from subprocess import check_output 
import pandas as pd
import numpy as np
import fitsio
from glob import glob
from collections import defaultdict
import json

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
  """Runs a bash comand, one that isnt meant to return anything"""
  print('UNIX cmd: %s' % cmd)
  if os.system(cmd): raise ValueError

def getbash(cmd):
  """Runs a bash command and returns what it would return
  
  Returns:
    string with \n as delimieters
  """
  #return check_output(["find", "doc","-name","*.png"]) 
  cmd_arr= cmd.split(' ')
  return check_output(cmd_arr) 

def writelist(lis,fn):
  if os.path.exists(fn):
    os.remove(fn)
  with open(fn,'w') as foo:
    for li in lis:
      foo.write('%s\n' % li)
  print('Wrote %s' % fn)
  if len(lis) == 0:
    print('Warning: %s is empty list' % fn) 

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

def merge_tables_fns(fn_list,textfile=True,
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


def _shrink_img(imgfn,imgfn_new, camera='decam'):
  """reads in imgfn and writes it to imgfn_new after removing most of the hdus"""
  assert(not 'project' in imgfn_new)
  hdu= fitsio.FITS(imgfn,'r')
  new= fitsio.FITS(imgfn_new,'rw')
  
  new.write(hdu[0].read(),header=hdu[0].read_header())
  if camera == 'decam':
    ccdnames= ['N4','S4', 'S22','N19']
  for ccdname in ccdnames:
     try: 
       data= hdu[ccdname].read()
       h= hdu[ccdname].read_header()
       new.write(data, extname=ccdname, header=h)
     except OSError:
       pass
  new.close()
  print('wrote %s' % imgfn_new)


def shrink_img(camera='decam'):
  root='/project/projectdirs/cosmo/staging/decam'
  images= ['DECam_CP/CP20150407/c4d_150409_000747_ooi_g_v1.fits.fz',
           'DECam_CP/CP20150407/c4d_150409_000424_ooi_r_v1.fits.fz',
           'DECam_CP/CP20150407/c4d_150409_001645_ooi_z_v1.fits.fz']
  for image in images:
    fn= os.path.join(root,image)
    # ooi
    fn_new='small_'+os.path.basename(fn) #.replace('.fz','')
    _shrink_img(fn, fn_new, camera=camera)
    # ood
    fn= fn.replace("oki","ood").replace('ooi','ood')
    fn_new= fn_new.replace("oki","ood").replace('ooi','ood')
    _shrink_img(fn, fn_new, camera=camera)


def add_fwhmcp_to_legacypipe_table(T, json_fn='json.txt'):
  """adds fwhm from CP header as a new colum for each ccd"""
  proj_fn='/project/projectdirs/cosmo/staging/'
  fwhm_cp= defaultdict(dict)
  fns= set(np.char.strip(T.image_filename))
  for cnt,fn in enumerate(fns):
    print('%d/%d' % (cnt+1,len(fns)))
    hdu= fitsio.FITS(proj_fn + fn)
    isImg= np.char.strip(T.image_filename) == fn
    for ccdname in np.char.strip(T[isImg].ccdname):
      h= hdu[ccdname].read_header()
      fwhm_cp[fn][ccdname]= h['FWHM'] * 0.262 
  import json
  json.dump(fwhm_cp, open(json_fn,'w'))
  print('Wrote %s' % json_fn)

def add_fwhmcp():
  root= '/global/cscratch1/sd/kaylanb/zpts_out/ebossDR5/'
  T=fits_table(root+"decam/merged_legacypipe_nocuts.fits.gz")
  
  #bad=T.copy()
  #bad.cut( np.isfinite(bad.fwhm) == False)
  #add_fwhmcp_to_legacypipe_table(bad, json_fn='fwhmcp_where_nan.json')
  
  good=T.copy()
  good.cut( np.isfinite(good.fwhm))
  #good= good[:500]
  add_fwhmcp_to_legacypipe_table(good, json_fn='fwhmcp_where_real.json')

def _insert_fwhmcp(T, json_fn):
  f= open(json_fn, 'r')
  fwhm_cp= json.loads(f.read())
  data= np.zeros(len(T)) -1
  for fn in fwhm_cp.keys():
    hasFn= np.char.strip(T.image_filename) == fn
    for ext in fwhm_cp[fn].keys():
      hasCCD= np.char.strip(T.ccdname) == ext
      data[(hasCCD) & (hasFn)]= fwhm_cp[fn][ext]
  T.set('fwhm_cp',data)
  return T

def replace_nans():
  T= fits_table("decam/merged_legacypipe_nocuts.fits.gz") 
  f= open('fwhmcp_where_nan.json', 'r')
  fwhm_cp= json.loads(f.read())
  data= T.fwhm
  fns= fwhm_cp.keys()
  for cnt,fn in enumerate(fns):
    print('%d/%d' % (cnt+1,len(fns)))
    hasFn= np.char.strip(T.image_filename) == fn
    for ext in fwhm_cp[fn].keys():
      hasCCD= np.char.strip(T.ccdname) == ext
      data[(hasCCD) & (hasFn)]= fwhm_cp[fn][ext] / 0.262
  T.set('fwhm',data)
  return T

def replace_all():
  T= replace_nans()
  T.writeto('merged_legacypipe_nocuts_replace_nans.fits')
  f= open('fwhmcp_where_real.json', 'r')
  fwhm_cp= json.loads(f.read())
  data= T.fwhm
  fns= fwhm_cp.keys()
  for cnt,fn in enumerate(fns):
    print('%d/%d' % (cnt+1,len(fns)))
    hasFn= np.char.strip(T.image_filename) == fn
    for ext in fwhm_cp[fn].keys():
      hasCCD= np.char.strip(T.ccdname) == ext
      data[(hasCCD) & (hasFn)]= fwhm_cp[fn][ext] / 0.262
  T.set('fwhm',data)
  T.writeto('merged_legacypipe_nocuts_replace_all.fits')

def insert_fwhmcp():
  # replace Nans with something
  leg=fits_table("decam/merged_legacypipe_nocuts.fits.gz") 
  #leg.cut(np.isfinite(leg.fwhm) == False)
  #return _insert_fwhmcp(leg,'fwhmcp_where_nan.json')  
  leg.cut(np.isfinite(leg.fwhm))
  leg=leg[:500]
  return _insert_fwhmcp(leg,'fwhmcp_where_real.json')  
