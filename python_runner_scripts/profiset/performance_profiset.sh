#!/bin/bash
#PBS -N profiset-GMM
#PBS -l select=1:ncpus=2:mem=320gb:cluster=elwe
#PBS -l walltime=36:00:00
#PBS -m ae

module add python-3.6.2-gcc
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1

source /storage/brno6/home/wollf/learned-indexes/venv/bin/activate
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes

python3 performance_profiset.py
