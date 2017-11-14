#! /bin/bash

# Example
# qdo launch obiwan 3 --cores_per_worker 4 --batchqueue debug --walltime 00:05:00 --script $obiwan_code/obiwan/bin/qdo_job_test.sh --keep_env

export camera=mosaic
export image_fn="$1"
export out_dir=/global/cscratch1/sd/dstn/psf-zpts-dr6
export cal_dir=/global/cscratch1/sd/dstn/dr6-calibs

# Redirect logs
export log=$out_dir/$(echo $image_fn | sed s+/+_+g | sed s/.fits.fz/.log/g)
mkdir -p $(dirname $log)
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export PYTHONPATH=~/astrometry:~/tractor:~/legacypipe/py:.:${PYTHONPATH}
echo "PYTHONPATH: $PYTHONPATH"

python legacyzpts/legacy_zeropoints.py \
	--camera ${camera} --image ${image_fn} --outdir ${out_dir} \
    --not_on_proj \
    --calibdir ${cal_dir} --splinesky --psf \
    --run-calibs \
    >> $log 2>&1
