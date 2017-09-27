#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --account=desi
#SBATCH -J merge
#SBATCH -L SCRATCH,project
#SBATCH -C haswell

export file_list=done_legacypipe.txt
export outname=merged_legacypipe.fits

# Load production env and env vars
source $CSCRATCH/zpts_code/legacyzpts/etc/modulefiles/bashrc_nersc
# check have new env vars
: ${zpts_code:?}

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}

# Redirect logs
export log=`echo $outname|sed s#.fits#.log#g`
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun -n $tasks -c 1 \
  python $zpts_code/legacyzpts/py/legacyzpts/legacy_zeropoints_merge.py 
  --file_list $file_list --nproc $tasks \
  --outname $outname \
    >> $log 2>&1
