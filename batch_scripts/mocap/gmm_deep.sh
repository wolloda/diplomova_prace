#!/bin/bash
#PBS -N mocap_gmm_deep.sh
#PBS -l select=1:ncpus=2:mem=140gb:cluster=alfrid
#PBS -l walltime=23:59:00

module add python-3.6.2-gcc
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1

source /storage/brno6/home/wollf/learned-indexes/venv/bin/activate
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/python_runner_scripts/mocap

python3  gmm_deep.py
