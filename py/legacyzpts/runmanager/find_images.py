"""Module to make the 'image_list.txt' file

"""
import numpy as np
import os

from legacyzpts.common import getbash,writelist

PROJA='/global/projecta/projectdirs/cosmo/staging/'
PROJ='/project/projectdirs/cosmo/staging/'
CAMERAS= ['decam','mosaic','90prime']
AKA= {'decam':'decam/DECam_CP',
      'mosaic':'mosaicz/MZLS_CP',
      '90prime':'bok/BOK_CP'}
      

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
    cmd= 'find %s -name %s*' % (PROJ + AKA[camera], name)
    out= getbash(cmd)
    if len(out) == 0:
      raise ValueError
  except ValueError:
    # getbash() failed OR succeeded but found nothing
    try: 
      cmd= 'find %s -name %s*fits.fz' % (PROJA + AKA[camera], name)
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
    out= get_abspath(imgname,camera=camera)
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
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--camera',default=None,help='',required=True)
  parser.add_argument('--image_list',default=None,help='text file listing c4*.fits.fz like image names',required=True)
  parser.add_argument('--outdir',default='.',help='',required=False)
  args = parser.parse_args()
 
  assert('.txt' in args.image_list)
  outfn= args.image_list.replace('.txt','_abs.txt')
  if not os.path.exists(outfn):
    images= np.loadtxt(args.image_list,dtype=str)
    fns= get_abspaths(images,camera=args.camera,
                      ignore_duplicates=True) 
    writelist(fns, outfn)
  else:
    print('Already exists %s' % outfn)


