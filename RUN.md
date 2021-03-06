# Run legacyzpts @ NERSC

### Code and Data
The code runs with the same python environment on either Cori or Edison, so install it to one place. Git clone the repos to a directory "zpts_code"
```sh
export zpts_code=$CSCRATCH/zpts_code
mkdir $zpts_code
cd $zpts_code
git clone https://github.com/legacysurvey/legacyzpts.git
git clone https://github.com/legacysurvey/legacypipe.git
```

### Python environment
The code runs using Ted Kisners [desiconda](https://github.com/desihub/desiconda.git) package for the imaging pipeline. We also need `qdo` for production runs. First create a duplicate conda environment of Teds build, then install qdo. You can either 

1)
source this bash script to use the environment on my scratch space
```sh
source $zpts_code/legacyzpts/etc/modulefiles/bashrc_nersc
qdo list
```
you should now see a list of the qdo queues

Or 

2)
install to your own scratch
```sh
module use /global/common/${NERSC_HOST}/contrib/desi/desiconda/20170818-1.1.12-img/modulefiles
module load desiconda/20170818-1.1.12-img
conda create --prefix $zpts_code/conda_envs/legacyzpts --file $DESICONDA/pkg_list.txt
```

Go get a coffee, this will take a few mins. When it finishes, install qdo
```sh
source activate $zpts_code/conda_envs/legacyzpts
cd $zpts_code
git clone https://bitbucket.org/berkeleylab/qdo.git
cd qdo
python setup.py install
cd ../
qdo list
```
you should now see a list of the qdo queues

### Run the test suite
Generate all the legacyzpts outputs for 3 DECam exposures (1 per band and just 2 CCDs), 1 Mosaic3 exposure, and 2 90Prime exposures (1 per band) 
```sh
cd $zpts_code/legacyzpts
#pytest tests/test_legacyzpts_runs.py
python tests/test_legacyzpts_runs.py
```

Once that completes, check that all computed quantities are very close to the matched values in the original IDL zeropoints files and in the survey-ccds files from DR3 and DR4.
```sh
#pytest tests/test_against_idl.py
#pytest tests/test_against_surveyccds.py
python tests/test_against_idl.py
python tests/test_against_surveyccds.py
```
That should print out a ton of info, showing the residuals in ccdzpt, skycounts, ccdnmatch, etc.

### Production Runs

First make lists of images to process 
```sh
find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*/k4m*ooi*.fits.fz > mosiac_images.txt
find /project/projectdirs/cosmo/staging/bok/BOK_CP/CP*/ksb*ooi*.fits.fz > 90prime_images.txt
```
then turn them into a list of QDO tasks
```sh
for camera in 90prime mosaic;do python $zpts_code/legacyzpts/py/legacyzpts/runmanager/qdo_tasks.py --camera ${camera} --imagelist ${camera}_images.txt;done
```
Note, youll need to do more than find the files to create the image lists. For example, cutting on date, selecting the right CP version of each file. This ipynb https://github.com/legacysurvey/legacyzpts/blob/master/doc/nb/image_list.ipynb shows how I did this for DR6.

Next well setup QDO. We use "qdo" to manage the thousands of production jobs, but for testing it is useful to have a script to run a single job. There are also two ways to set up the compute node environment: the conda environment above or a Docker Image. This gives us 
 1) Single job
 * A) conda environment
 * B) Docker image
 2) Many jobs (qdo)
 * A) conda environment
 * B) Docker image

#### 1A)
Run a `Serial job`, see
https://github.com/legacysurvey/legacyzpts/blob/master/bin/slurm_job.sh

```sh
export name_for_run=ebossDR5
export outdir=$zpts_out/$name_for_run
mkdir -p $outdir
cd $outdir
cp $zpts_code/legacyzpts/bin/slurm_job.sh ./
```

Edit these lines for your run:
```sh
export camera=decam
export name_for_run=ebossDR5
export proj_dir=/project/projectdirs/cosmo/staging
export scr_dir=/global/cscratch1/sd/kaylanb/zpts_out/${name_for_run}
export image_list=$scr_dir/image_list.txt
```

then Submit the job
```sh
sbatch slurm_job.sh
```

Now you could run the MPI version, see 
https://github.com/legacysurvey/legacyzpts/blob/master/bin/slurm_job_mpi.sh
but Ill continue on to qdo

To submit the MPI version for all exposures, say from night CP20160225, do
```
cd $zpts_out/$name_for_run
night=CP20160225
find $proj_dir/decam/DECam_CP/${night}/c4d*oki*.fits.fz > image_list.txt
sbatch $zpts_code/legacyzpts/bin/slurm_job_mpi.sh
```

#### 1B)
Coming soon

#### 2A)
Qdo

Make a queue for your run and upload all tasks for all cameras to it
```
qdo create zpts_eBOSS
for camera in decam mosaic 90prime;do qdo load zpts_eBOSS ${camera}_tasks.txt;done
```

Get the job script, see
https://github.com/legacysurvey/obiwan/blob/master/bin/qdo_job.sh
```sh
cd $zpts_out/$name_for_run
cp $zpts_code/legacyzpts/bin/qdo_job.sh ./
```

Edit these lines:
```sh
export name_for_run=ebossDR5
```

Now launch qdo workers
```sh
cd $zpts_out/$name_for_run
qdo launch zpts_eBOSS 320 --cores_per_worker 1 --batchqueue debug --walltime 00:30:00 --script $zpts_out/$name_for_run/qdo_job.sh --keep_env
```

#### 2B)
Coming soon

### TODO

* --night option for legacy_zeropoints, which will run all exposures for that night 

### Managing your qdo production run
There are two key scripts for inspecting the outputs, they are in `legacyzpts/py/legacyzpts/runmanager/`
 1) legacyzpts/py/legacyzpts/runmanager/status.py 
  * lists log files for QDO succeeded, failed, and running jobs
  * parses log files for failed jobs for known failure modes and remakes lists for associated modes
  * add new regex commands to this script as you document new failure modes
 2) legacyzpts/py/legacyzpts/runmanager/run_ccd.py
  * runs a single CCD for an image for a given camera
  * you can find some error you dont understand in the log*.txt files, then use run_ccd.py to reproduce it

Manage your qdo production run with `legacyzpts/py/legacyzpts/runmanager/status.py`. List the log files associated with each QDO state "succeeded, failed, running", and list the errors in each log file with
```sh
export name_for_run=ebossDR5
cd $zpts_code/$name_for_run
python $zpts_code/legacyzpts/py/legacyzpts/runmanager/status.py --qdo_quename zpts_eBOSS --outdir "$zpts_out/$name_for_run"
```

Any "running" jobs that remain after all qdo jobs have finished are time outs. They get miscomunicated to the QDO db and need to be rerun. Resubmit them and delete the associated outputs for that task with 
```sh
python $zpts_code/legacyzpts/py/legacyzpts/runmanager/status.py --qdo_quename zpts_eBOSS --outdir $zpts_code/$name_for_run --running_to_pending --modify
```
Note run without the "--modify" to see what tasks would be reset and file removed

The types of errors found in the "failed" log files are listed. You can rerun all tasks and remove associated outputs specifying the name of the error. For example, "log not exist"
```sh
python $zpts_code/legacyzpts/py/legacyzpts/runmanager/status.py --qdo_quename zpts_eBOSS --outdir $zpts_code/$name_for_run --failed_message_to_pending "log not exist" --modify
```

### QA on the existing tables
Merge the tables youve made so far and do QA on them. The `*-zpt.fits` table has the majority of information so lets start there
```sh
export file_list=done_zpt.txt
find $zpts_out/$name_for_run/decam -name "*-zpt.fits" > $file_list
export outname=merged_zpt.fits
python $zpts_code/legacyzpts/py/legacyzpts/legacy_zeropoints_merge.py --file_list $file_list --nproc 1 --outname $outname
```

How many ccds have an err message? What are the err messages that occured? What does the zeropoint distribution or ra and dec offsets?
```sh
python $zpts_code/legacyzpts/py/legacyzpts/runmanager/qa.py --camera decam --zptfn $outname
```

### Final merged table
When all runs are finished merging the thousands of tables could take a while, so 
do it in MPI
```sh
cd $zpts_code/$name_for_run
cp $zpts_out/legacyzpts/bin/slurm_job_merge.sh .
```
Edit
```sh
export file_list=done_legacypipe.txt
find $zpts_out/$name_for_run/decam -name "*-legacypipe.fits" > $file_list
export outname=merged_legacypipe.fits
```
then run with
```sh
sbatch slurm_job_merge.sh
```

### Oringal Instructions

1 generate file list of cpimages, e.g. for everything mzls. For example,
```
find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*v2/k4m*ooi*.fits.fz > mosaic_allcp.txt
```
2 use batch script "submit_zpts.sh" to run legacy-zeropoints.py
 * scaling is good with 10,000+ cores
 * key is Yu Feng's bcast, we made tar.gz files for all the NERSC HPCPorts modules and Yu's bcast efficiently copies them to ram on every compute node for fast python startup
 * the directory containing the tar.gz files is /global/homes/k/kaylanb/repos/yu-bcase
 * also on kaylans tape: bcast_hpcp.tar
3 queus
 * debug queue gets all zeropoints done < 30 min
 * set SBATCH -N to be as many nodes as will give you mpi tasks = nodes*cores_per_nodes ~ number of cp images
 * ouput ALL plots with --verboseplots, ie Moffat PSF fits to 20 brightest stars for FWHM
 4 Make a file (e.g. zpt_files.txt) listing all the zeropoint files you just made (not includeing the -star.fits ones), then 
  * compare legacy zeropoints to Arjuns 
  ```
  python legacy-zeropoints.py --image_list zpt_files.txt --compare2arjun
  ```
  * gather all zeropoint files into one fits table
  ```python legacy-zeropoints-gather.py --file_list zpt_files.txt --nproc 1 --outname gathered_zpts.fits
  ```


