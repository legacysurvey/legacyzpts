#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J zpts
#SBATCH -L SCRATCH,project
#SBATCH -C haswell

export hardcores=8
export camera=decam
export name_for_run=dr6
export scr_dir=/global/cscratch1/sd/kaylanb/zpts_out/${name_for_run}

export image_list=image_list.txt
let softcores=2*${hardcores}

# Load production env and env vars
source $CSCRATCH/zpts_code/legacyzpts/etc/modulefiles/bashrc_nersc
# check have new env vars
: ${zpts_code:?}

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}/${hardcores}

# Redirect logs
export log=`echo $image_fn|sed s#${proj_dir}#${scr_dir}#g`
mkdir -p $(dirname $log)
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd $zpts_code/legacyzpts/py
echo tasks=$tasks softcores=${softcores} hardcores=${hardcores}
srun -n $tasks -c ${softcores} \
	python legacyzpts/legacy_zeropoints_mpiwrapper.py \
    --camera ${camera} --image_list ${image_list} \
    --outdir ${scr_dir} --nproc $tasks
