#! /bin/bash

# Example
# qdo launch obiwan 3 --cores_per_worker 4 --batchqueue debug --walltime 00:05:00 --script $obiwan_code/obiwan/bin/qdo_job_test.sh --keep_env

export camera=decam
export name_for_run=ebossDR5
export proj_dir=/project/projectdirs/cosmo/staging
export scr_dir=/global/cscratch1/sd/kaylanb/zpts_out/${name_for_run}
export image_fn="$1"

# Load production env and env vars
source $CSCRATCH/zpts_code/legacyzpts/etc/modulefiles/bashrc_nersc
# check have new env vars
: ${zpts_code:?}

# Redirect logs
export log=`echo $image_fn|sed s#${proj_dir}#${scr_dir}#g|sed s#.fits.fz#.log#g`
mkdir -p $(dirname $log)
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export threads=1

cd $zpts_code/legacyzpts/py
python legacyzpts/legacy_zeropoints.py \
	--camera ${camera} --image ${image_fn} --outdir ${scr_dir} \
  --ps1_only \
    >> $log 2>&1


