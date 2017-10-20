import qdo
import os
import numpy as np
from glob import glob
import re
from collections import defaultdict

from legacyzpts.common import dobash

QDO_RESULT= ['running', 'succeeded', 'failed']

def projfn_to_scrfn(proj_fn,outdir):
  """Converts project fn like to scratch fn
  Args:
    proj_fn: abs path to image on project
    outdir: where slurm-*.out files are
  """
  assert('staging' in proj_fn)
  return os.path.join(outdir, 
                      proj_fn.split('staging/')[1])

def get_logfile(proj_fn,outdir):
  """
  Args:
    proj_fn: abs path to image on project
    outdir: where slurm-*.out files are
  """
  return projfn_to_scrfn(proj_fn,outdir).replace('.fits.fz','.log')


def get_slurm_files(outdir):
  return glob( outdir + '/slurm-*.out')

def writelist(lis,fn):
  if os.path.exists(fn):
    os.remove(fn)
  with open(fn,'w') as foo:
    for li in lis:
      foo.write('%s\n' % li)
  print('Wrote %s' % fn)
  if len(lis) == 0:
    print('Warning: %s is empty list' % fn) 


class QdoList(object):
  """Queries the qdo db and maps log files to tasks and task status
  
  Args:
    outdir: obiwan outdir, the slurm*.out files are there
    que_name: ie. qdo create que_name
    skip_suceeded: number succeeded tasks can be very large for production runs, 
      this slows down code so skip those tasks
  """

  def __init__(self,outdir,que_name='zpts_eBOSS',skip_suceeded=False):
    self.outdir= outdir
    self.que_name= que_name
    self.skip_suceeded= skip_suceeded

  def get_tasks_logs(self):
    """get tasks, logs for the three types of qdo status
    Running, Succeeded, Failed"""
    # Logs for all Failed tasks
    tasks={}
    ids={}
    logs= defaultdict(list)
    print('qdo Que: %s' % self.que_name)
    q = qdo.connect(self.que_name)
    for res in QDO_RESULT:
      if self.skip_suceeded and res == 'succeeded':
        continue
      print('listing %s tasks' % res.upper())
      # List of "brick rs" for each QDO_RESULT  
      tasks[res] = [a.task 
                    for a in q.tasks(state= getattr(qdo.Task, res.upper()))]
      ids[res] = [a.id 
                    for a in q.tasks(state= getattr(qdo.Task, res.upper()))]
      # Corresponding log, slurm files 
      print('logfiles for %s tasks' % res.upper())
      for task in tasks[res]:
        if len(task.split(' ')) == 2:
          camera,projfn = task.split(' ')
        else:
          projfn = task.split(' ')[0]
        # Logs
        logfn= get_logfile(projfn, self.outdir)
        logs[res].append( logfn )
    return tasks,ids,logs

  def rerun_tasks(self,task_ids, modify=False):
    """set qdo tasks state to Pending for these task_ids
    
    Args:
      modify: True to actually reset the qdo tasks state AND to delete
      all output files for that task
    """
    q = qdo.connect(self.que_name)
    for task_id in task_ids:
      try:
        task_obj= q.tasks(id= int(task_id))
        camera,projfn = task_obj.task.split(' ')
        logfn= get_logfile(projfn, self.outdir)
        rmcmd= "rm %s*" % logfn.replace('.log',"")
        if modify:
          task_obj.set_state(qdo.Task.PENDING)
          dobash(rmcmd)
        else:
          print('would remove id=%d, which corresponds to taks_obj=' % task_id,task_obj)
          print('would call dobash(%s)' % rmcmd)
      except ValueError:
        print('cant find task_id=%d' % task_id)

class RunStatus(object):
  """Tallys which QDO_RESULTS actually finished, what errors occured, etc.
  
  Args:
    tasks: dict, each key is list of qdo tasks
    logs: dict, each key is list of log files for each task

  Defaults:
    regex_errs: list of regular expressions matching possible log file errors
  """
    
  def __init__(self,tasks,logs):
    self.tasks= tasks
    self.logs= logs
    self.regex_errs= [
        'MemoryError',
        "FAIL: All stars elimated after 2nd round of cuts",
        "NameError: name 'imgfn_proj' is not defined",
        "ValueError: Inconsistent data column lengths: {0, 1}",
        "Photometry on 0 stars",
        "OSError: File not found: '/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-",
        r'Could\ not\ find\ fwhm_cp.*?FWHM',
        r'Could\ not\ find\ fwhm_cp.*?SEEINGP1',
        r'unknown\ record:\ PROPID',
        r'File\ not\ found:.*?ood.*?fits.fz',
        r'k4m.*?_ooi_gd',
        r'k4m.*?_ooi_rd',
        r'k4m.*?_ooi_wrc4',
        r'TAN\ header:\ expected\ CTYPE1\ =\ RA---TAN,.*got\ CTYPE1\ =\ "RA---ZPX"',
        ]
    self.regex_badexp= [
        r'CP20170204\/ksb_170205_114709_ooi_r_v1\.fits\.fz',
        r'CP20160226v2\/k4m_160227_115515_ooi_zd_v2\.fits.fz'
        ]
    self.regex_errs_extra= ['Other','log not exist','BadExposure']

  def get_tally(self):
    tally= defaultdict(list)
    for res in self.tasks.keys():
      if res == 'succeeded':
        for log in self.logs[res]:
          with open(log,'r') as foo:
            text= foo.read()
          if (("Wrote 2 stars tables" in text) &
              ("Done" in text)):
            tally[res].append( 1 )
          else:
            tally[res].append( 0 )
      elif res == 'running':
        for log in self.logs[res]:
          tally[res].append(1)
      elif res == 'failed':
        for log in self.logs[res]:
          if not os.path.exists(log):
            tally[res].append( 'log not exist')
            continue
          with open(log,'r') as foo:
            text= foo.read()
          found_err= False
          for regex in self.regex_errs:
            foundIt= re.search(regex, text)
            if foundIt:
              tally[res].append(regex)
              found_err=True
              break
          if not found_err:
            for regex in self.regex_badexp:
              foundIt= re.search(regex, text)
              if foundIt:
                tally[res].append('BadExposure')
                found_err=True
                break
          if not found_err:
            tally[res].append('Other')
          
    # numpy array, not list, works with np.where()
    for res in tally.keys():
      tally[res]= np.array(tally[res])
    return tally

  def print_tally(self,tally):
    for res in self.tasks.keys():
      print('--- Tally %s ---' % res)
      if res == 'succeeded':
         print('%d/%d = done' % (len(tally[res]), np.sum(tally[res])))
      elif res == 'failed':
        for regex in self.regex_errs + self.regex_errs_extra:
          print('%d/%d = %s' % (
                   np.where(tally[res] == regex)[0].size, len(tally[res]), regex))
      elif res == 'running':
         print('%d/%d : need rerun' % (len(tally[res]),len(tally[res])))
  
  def get_logs_for_failed(self,regex='Other'):
    """Returns log and slurm filenames for failed tasks labeled as regex"""
    return self.logs[ tally['failed'] == regex ]



if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--qdo_quename',default='zpts_eBOSS',help='',required=False)
  parser.add_argument('--outdir',default='/global/cscratch1/sd/kaylanb/zpts_out/ebossDR5',help='',required=False)
  parser.add_argument('--skip_suceeded',action='store_true',default=False,help='number succeeded tasks can be very large for production runs and slows down status code',required=False)
  parser.add_argument('--running_to_pending',action="store_true",default=False,help='set to reset all "running" jobs to "pending"')
  parser.add_argument('--failed_message_to_pending',action='store',default=None,help='set to message of failed tak and reset all failed tasks with that message to pending')
  parser.add_argument('--modify',action='store_true',default=False,help='set to actually reset the qdo tasks state AND to delete IFF running_to_pending or failed_message_to_pending are set')
  args = parser.parse_args()
  print(args)

  Q= QdoList(args.outdir,que_name=args.qdo_quename,
             skip_suceeded=args.skip_suceeded)
  tasks,ids,logs= Q.get_tasks_logs()
  
  # Write log fns so can inspect
  for res in logs.keys():
    writelist(logs[res],"%s_%s_logfns.txt" % (args.qdo_quename,res))
  R= RunStatus(tasks,logs)
  tally= R.get_tally()
  R.print_tally(tally)

  #err_logs= R.get_logs_for_failed(regex='Other')
  for err_key in R.regex_errs + R.regex_errs_extra:
    err_logs= np.array(logs['failed'])[ tally['failed'] == err_key ]
    err_tasks= np.array(tasks['failed'])[ tally['failed'] == err_key ]
    err_string= ((err_key[:10] + err_key[-10:])
                 .replace(" ","_")
                 .replace("/","")
                 .replace("*","")
                 .replace("?","")
                 .replace(":",""))
    writelist(err_logs,"logs_%s_%s.txt" % (args.qdo_quename,err_string))
    writelist(err_tasks,"tasks_%s_%s.txt" % (args.qdo_quename,err_string))

  if args.running_to_pending:
    if len(ids['running']) > 0:
      Q.rerun_tasks(ids['running'], modify=args.modify)
  if args.failed_message_to_pending:
    hasMessage= np.where(tally['failed'] == args.failed_message_to_pending)[0]
    if hasMessage.size > 0:
      theIds= np.array(ids['failed'])[hasMessage]
      Q.rerun_tasks(theIds, modify=args.modify)


