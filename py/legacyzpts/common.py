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

from legacyzpts.legacy_zeropoints import get_90prime_expnum 

try:
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.libkd.spherematch import match_radec
except ImportError:
    pass

CAMERAS=['decam','mosaic','90prime']

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

def writejson(d,fn):
    """Save a dict to json file"""
    json.dump(d, open(fn,'w'))
    print('Wrote json %s' % fn)

def loadjson(fn):
    """Retuns a dict from reading json file"""
    f= open(fn, 'r')
    return json.loads(f.read())

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


def _shrink_img(imgfn,imgfn_new, ccdnames=[]):
    """reads in imgfn and writes it to imgfn_new after removing most of the hdus"""
    assert(not 'project' in imgfn_new)
    assert(len(ccdnames) > 0)
    hdu= fitsio.FITS(imgfn,'r')
    new= fitsio.FITS(imgfn_new,'rw')

    new.write(hdu[0].read(),header=hdu[0].read_header())
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
    if camera == 'decam':
        root='/project/projectdirs/cosmo/staging/decam'
        images= ['DECam_CP/CP20150407/c4d_150409_000747_ooi_g_v1.fits.fz',
                 'DECam_CP/CP20150407/c4d_150409_000424_ooi_r_v1.fits.fz',
                 'DECam_CP/CP20150407/c4d_150409_001645_ooi_z_v1.fits.fz']
        ccdnames=['N4','S4', 'S22','N19']
    for image in images:
        fn= os.path.join(root,image)
        # ooi
        fn_new='small_'+os.path.basename(fn) #.replace('.fz','')
        _shrink_img(fn, fn_new, ccdnames=ccdnames)
        # ood
        fn= fn.replace("oki","ood").replace('ooi','ood')
        fn_new= fn_new.replace("oki","ood").replace('ooi','ood')
        _shrink_img(fn, fn_new, ccdnames=ccdnames)

def ccds_touching_bricks(bricks,ccds,camera,
                         forcesep=None):
    """

    Args:
        bricks: bricks fits table
        ccds: ccds fits table
        camera:

    Returns:
        bricks,ccds: tuple of fits tables of bricks and ccds that are touching
    """
    assert(camera in CAMERAS)
    bricksize = 0.25
    maxSideArcsec= {'decam':0.262* 4094,
                  'mosaic':0.262* 4096,
                  '90prime':0.455* 4096}
    # A bit more than 0.25-degree brick radius + image radius
    search_radius = 1.05 * np.sqrt(2.) * (bricksize +
                                        (maxSideArcsec[camera] / 3600.))/2.
    if forcesep:
        search_radius= forcesep
    I,J,d = match_radec(bricks.ra, bricks.dec, ccds.ra, ccds.dec, search_radius,
                      nearest=True)
    lenB,lenC= len(bricks),len(ccds)
    bricks.cut(I)
    ccds.cut(J)
    print('%d/%d bricks, %d/%d ccds are touching' % 
          (len(bricks),lenB,len(ccds),lenC))
    return bricks,ccds 

def ccds_for_brickname(brickname,ccds,camera,forcesep=None):
    """get list of ccds touching a brick, brick is specified by string name"""
    bricks= fits_table('/global/project/projectdirs/cosmo/data/legacysurvey/dr4/survey-bricks.fits.gz')
    bricks.cut(np.char.strip(bricks.brickname) == brickname)
    _,c= ccds_touching_bricks(bricks,ccds,camera,forcesep)
    return c
  

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
    leg=leg[map_90prime_images_to_expnum:500]
    return _insert_fwhmcp(leg,'fwhmcp_where_real.json')  


def map_90prime_images_to_expnum(imagelist):
    """reads 90prime headers and returns expnums

    Args:
        imagelist: text file listing abs path to 90prime files

    Returns:
        dict
    """
    fns= np.loadtxt(imagelist,dtype=str)
    expnum= {get_90prime_expnum(
            fitsio.read_header(fn, 0)):os.path.join(fn.split('/')[-2],fn.split('/')[-1])
           for fn in fns}
    with open('all_expnum_img.txt','w') as foo:
        for key in expnum.keys(): 
            foo.write('%s %s\n' % (key,expnum[key]))
    print('Wrote all_expnum_img.txt')
    return expnum

def fall2015_90prime_images(tiles_fn,imageliset):
    """returns science-able 90prime Fall 2015 images given bass_tiles file and bass image list"""
    tiles= fits_table(tiles_fn)
    g_tiles= tiles[((tiles.g_date == '2015-11-12') |
                  (tiles.g_date == '2015-11-13'))]
    r_tiles= tiles[((tiles.r_date == '2015-11-12') |
                  (tiles.r_date == '2015-11-13'))]
    expnums= list(g_tiles.g_expnum) + list(r_tiles.r_expnum)
    exp2img= map_90prime_images_to_expnum(imageliset)
    final={}
    for expnum in expnums:
        if expnum in exp2img.keys():
            final[expnum]= exp2img[expnum]
        else:
            print('%s not in exp2img' % expnum)
    return final


def run():
    #map_90prime_images_to_expnum('/global/cscratch1/sd/kaylanb/images.txt')
    final= fall2015_90prime_images('/global/cscratch1/sd/kaylanb/svn_90prime/obstatus/bass-tiles_obstatus.fits','/global/cscratch1/sd/kaylanb/90prime_fall2015.txt')
    for key in final.keys(): print(key,final[key])
    with open('final_expnum_img.txt','w') as foo:
        for key in final.keys(): 
            foo.write('%s %s\n' % (key,final[key]))
    print('Wrote final_expnum_img.txt')






