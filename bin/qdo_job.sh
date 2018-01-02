#! /bin/bash

# Example
# qdo launch dr6_zpts 32 --cores_per_worker 1 --batchqueue debug --walltime 00:30:00 --script $obiwan_code/obiwan/bin/qdo_job_test.sh --keep_env

export name_for_run=dr6
export proj_dir=/project/projectdirs/cosmo/staging
export proja_dir=/global/projecta/projectdirs/cosmo/staging
export scr_dir=/global/cscratch1/sd/kaylanb/zpts_out/${name_for_run}
export camera="$1"
export image_fn="$2"

# Load production env and env vars
source $CSCRATCH/zpts_code/legacyzpts/etc/modulefiles/bashrc_nersc
# check have new env vars
: ${zpts_code:?}

# Redirect logs
export log=`echo $image_fn|sed s#${proj_dir}#${scr_dir}#g|sed s#${proja_dir}#${scr_dir}#g|sed s#.fits.fz#.log#g`
mkdir -p $(dirname $log)
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export threads=1

# Limit memory to avoid 1 srun killing whole node
if [ "$NERSC_HOST" = "edison" ]; then
    # 62 GB / Edison node = 65000000 kbytes
    maxmem=65000000
    let usemem=${maxmem}*${threads}/24
else
    # 128 GB / Cori node = 65000000 kbytes
    maxmem=134000000
    let usemem=${maxmem}*${threads}/32
fi
echo BEFORE
ulimit -Sa
ulimit -Sv $usemem
echo AFTER
ulimit -Sa


cd $zpts_code/legacyzpts/py
python legacyzpts/legacy_zeropoints.py \
	--camera ${camera} --image ${image_fn} --outdir ${scr_dir} \
    >> $log 2>&1


