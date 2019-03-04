#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 00:30:00
#SBATCH -A desi
#SBATCH -J zpts
#SBATCH -L SCRATCH,project
#SBATCH -C haswell

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Load the software we need
desiconda_version=20180512-1.2.5-img
module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles
module load desiconda

export LEGACYPIPE_DIR=$HOME/repos/git/legacypipe
export LEGACYZPTS_DIR=$HOME/repos/git/legacyzpts

source $LEGACYPIPE_DIR/bin/legacypipe-env

export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LEGACYZPTS_DIR/py:$PYTHONPATH

export PATH=$HOME/repos/build/$NERSC_HOST/bin:$PATH
export PYTHONPATH=$HOME/repos/build/$NERSC_HOST/lib/python3.6/site-packages:$PYTHONPATH

export camera=decam
export topdir=/global/u2/i/ioannis/dr8-calibs
export calibdir=$topdir/calib/$camera
export logdir=$topdir/log/$camera
export outdir=$topdir/zpts/$camera
export image_list=$topdir/$camera-image-list.txt

#export name_for_run=dr6
#export hardcores=8
#export image_list=image_list.txt
#let softcores=2*${hardcores}

#if [ "$NERSC_HOST" == "cori" ]; then
#    cores=32
#elif [ "$NERSC_HOST" == "edison" ]; then
#    cores=24
#fi
#let tasks=${SLURM_JOB_NUM_NODES}*${cores}/${hardcores}

export cores=16
export tasks=4

# Redirect logs
#export log=`echo $image_fn|sed s#${proj_dir}#${scr_dir}#g`
#mkdir -p $(dirname $log)
#echo Logging to: $log

#cd $zpts_code/legacyzpts/py
#echo tasks=$tasks softcores=${softcores} hardcores=${hardcores}

srun -n $tasks -c ${cores} python $LEGACYZPTS_DIR/py/legacyzpts/legacy_zeropoints_mpiwrapper.py \
    --camera ${camera} --image_list ${image_list} --calibdir ${calibdir} \
    --outdir ${outdir} --logdir ${logdir} --run-calibs --threads ${cores} \
    --nproc $tasks
