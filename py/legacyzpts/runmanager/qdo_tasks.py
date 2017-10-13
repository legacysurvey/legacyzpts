import os
import numpy as np

from legacyzpts.common import writelist

CAMERAS=['90prime','decam','mosaic']

class TaskList(object):
  def qdo_tasklist(self,camera,imagelist): 
    """writes qdo task list for a given camera and image list
   
    Returns:
     Writes qdo task list to file
    """
    assert(camera in CAMERAS)
    images= np.loadtxt(imagelist,dtype=str)
    # Tasks: camera imagefn
    tasks= ['%s %s' % (camera,fn) 
            for fn in images]
    writelist(tasks, 'tasks_%s.txt' % camera)


if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--camera', type=str, choices=['decam','mosaic','90prime'],required=True)
  parser.add_argument('--imagelist', type=str, required=True)
  args = parser.parse_args()

  TaskList().qdo_tasklist(args.camera,args.imagelist)

