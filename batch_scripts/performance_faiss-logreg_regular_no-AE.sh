#!/bin/bash
#PBS -N performance_faiss-logreg_regular_no-AE.sh
#PBS -l select=1:ncpus=2:mem=260gb:ngpus=1 -q gpu_long
#PBS -l walltime=48:00:00

source activate /storage/plzen1/home/tslaninakova/.conda/envs/tensorflow
export PATH=/storage/plzen1/home/tslaninakova/.conda/envs/tensorflow/bin:/software/anaconda3-4.0.0/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:/software/meta-utils/public:/usr/bin
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/

python3  performance_faiss-logreg_regular_no-AE.py
