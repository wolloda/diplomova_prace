#!/bin/bash
#PBS -N gmm_wide.sh
#PBS -l select=1:ncpus=2:mem=90gb:cluster=alfrid
#PBS -l walltime=24:00:00

module add python-3.6.2-gcc
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1

source /storage/brno6/home/wollf/learned-indexes/venv/bin/activate
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/python_runner_scripts/cophir

python3 gmm_wide.py
