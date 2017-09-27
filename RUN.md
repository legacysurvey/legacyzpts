# Run legacyzpts @ NERSC

### Code and Data
The code runs with the same python environment on either Cori or Edison, so install it to one place. Git clone the repos to a directory "zpts_code"
```sh
export zpts_code=$CSCRATCH/zpts_code
mkdir $zpts_code
cd $zpts_code
git clone https://github.com/legacysurvey/legacyzpts.git
git clone https://github.com/legacysurvey/legacypipe.git
cd legacypipe
git fetch
git checkout dr5_wobiwan
git checkout 5134bc49240ba py/legacyanalysis/ps1cat.py
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

### Unit tests
Run the tests with
```sh
cd $zpts_code/legacyzpts
pytest tests/
```
which should print "3 passed in <blah> seconds"

### Production Runs

We use "qdo" to manage the thousands of production jobs, but for testing it is useful to have a script to run a single job. There are also two ways to set up the compute node environment: the conda environment above or a Docker Image. This gives us 
 1) Single job
 * A) conda environment
 * B) Docker image
 2) Many jobs (qdo)
 * A) conda environment
 * B) Docker image

#### 1A)
Make a text file listing all the absolute paths to the CP image filenames you want to run on.
```
find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*v2/k4m*ooi*.fits.fz > image_list.txt
```
Ill use eBOSS DR5 PS1 only zeropoints as an example. Anand gave me the list of CP images, so I renamed it to image_list.txt

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

Make a new queue. 
```
qdo create zpts_eBOSS
qdo load zpts_eBOSS image_list.txt
```

Get the job script, see
https://github.com/legacysurvey/obiwan/blob/master/bin/qdo_job.sh
```sh
cd $zpts_out/$name_for_run
cp $zpts_code/legacyzpts/bin/qdo_job.sh ./
```

Edit these lines:
```sh
export camera=decam
export name_for_run=ebossDR5
export proj_dir=/project/projectdirs/cosmo/staging
export scr_dir=/global/cscratch1/sd/kaylanb/zpts_out/${name_for_run}
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
Manage your qdo production run with `legacyzpts/py/legacyzpts/runmanager/status.py`. List the log files associated with each QDO state "succeeded, failed, running", and list the errors in each log file with
```sh
cd $zpts_code/$name_for_run
python $zpts_code/legacyzpts/py/legacyzpts/runmanager/status.py --qdo_quename zpts_eBOSS --outdir $zpts_code/$name_for_run
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


