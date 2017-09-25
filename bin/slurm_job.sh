#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --account=desi
#SBATCH -J zpts
#SBATCH -L SCRATCH,project
#SBATCH -C haswell

export camera=decam
export name_for_run=ebossDR5
export proj_dir=/project/projectdirs/cosmo/staging
export scr_dir=/global/cscratch1/sd/kaylanb/zpts_out/${name_for_run}
export image_list=$scr_dir/image_list.txt

usecores=1
export image_fn=`head ${image_list} -n 1`

# Load production env and env vars
source $CSCRATCH/zpts_code/legacyzpts/etc/modulefiles/bashrc_nersc
# check have new env vars
: ${zpts_code:?}

# Redirect logs
export log=`echo $image_fn|sed s#${proj_dir}#${scr_dir}#g`
mkdir -p $(dirname $log)
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
#export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=$threads

cd $zpts_code/legacyzpts/py
srun -n 1 -c ${usecores} \
	python legacyzpts/legacy_zeropoints.py \
	--camera ${camera} --image ${image_fn} --outdir ${scr_dir} \
    >> $log 2>&1
