#!/bin/bash

#PBS -N plot_L1_distribution_60-epochs
#PBS -l select=ncpus=1:scratch_local=50mb
#PBS -l walltime=0:03:00

module add python36-modules-gcc

DATADIR=/storage/brno6/home/wollf/learned-indexes/learned-indexes/buckets_per_epoch_count/60_epochs 

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

cp $DATADIR/../plot_distribution.py $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp $DATADIR/L1_buckets.csv $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $SCRATCHDIR

python3 plot_distribution.py || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

cp L1_60_epochs.png $DATADIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch

