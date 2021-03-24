#!/bin/bash
#PBS -N performance_kmeans_deep_no-AE.sh
#PBS -l select=1:ncpus=2:mem=260gb
#PBS -l walltime=48:00:00

source activate /storage/plzen1/home/tslaninakova/.conda/envs/tensorflow
export PATH=/storage/plzen1/home/tslaninakova/.conda/envs/tensorflow/bin:/software/anaconda3-4.0.0/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:/software/meta-utils/public:/usr/bin
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/

python3  performance_kmeans_deep_no-AE.py