"""Module to make the 'image_list.txt' file

"""
import numpy as np
import os

from legacyzpts.common import getbash,writelist

PROJA='/global/projecta/projectdirs/cosmo/staging/'
PROJ='/project/projectdirs/cosmo/staging/'
CAMERAS= ['decam','mosaic','90prime']

def isCamera(camera):
  return camera in CAMERAS

def get_abspath(imgname,camera='decam'):
  """return abs path on project or projecta to image name

  Args:
    imgname: like 'c4d_150410_001704_ooi_g_v1.fits'

  Returns:
    byte char array with \n delimeters
  """
  assert(isCamera)
  name= imgname.replace(".fz","").replace(".fits","")
  try: 
    cmd= 'find %s -name %s*' % (PROJ + camera, name)
    out= getbash(cmd)
    if len(out) == 0:
      raise ValueError
  except ValueError:
    # getbash() failed OR succeeded but found nothing
    try: 
      cmd= 'find %s -name %s*fits.fz' % (PROJA + camera, name)
      out= getbash(cmd)
    except ValueError:
      raise ValueError('imgname doesnt exist %s' % imgname)
  return out

def get_abspaths(imgname_list,camera='decam',
                 ignore_duplicates=False):
  """return abs path on project or projecta to image name

  Args:
    imgname: like 'c4d_150410_001704_ooi_g_v1.fits'
  """

  fns= []
  for cnt,imgname in enumerate(imgname_list):
  #for cnt,imgname in enumerate(['c4d_130220_043932_ooi_z_a1.fits']):
    if cnt % 10 == 0: 
      print("Found %d/%d" % (cnt+1,len(imgname_list)))
    out= get_abspath(imgname,camera='decam')
    out= out.decode('utf-8') 
    fn_list= out.split('\n')
    if len(fn_list[1]) != 0:
      print("WARNING: multiple fns fo imgname=%s: " % imgname,fn_list)
      if ignore_duplicates:
        pass
      else:
        raise ValueError()
    fns.append( fn_list[0] )
  assert(len(fns) == len(imgname_list))
  return fns


if __name__ == "__main__":
  dr= '/global/cscratch1/sd/kaylanb/zpts_out/cosmos_dessn'
  for fn in ['COSMOS_decals-filelist.txt',
             'DESSNX3_decals-filelist.txt']:
    outfn= os.path.join(dr,fn).replace('.txt','-abs.txt')
    if not os.path.exists(outfn):
      images= np.loadtxt(os.path.join(dr,fn),dtype=str)
      fns= get_abspaths(images,camera='decam',
                        ignore_duplicates=True) 
      writelist(fns, outfn)
    else:
      print('Already exists %s' % outfn)


