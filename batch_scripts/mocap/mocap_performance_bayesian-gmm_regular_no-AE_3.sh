#!/bin/bash
#PBS -N mocap_performance_bayesian-gmm_regular_no-AE.sh
#PBS -l select=1:ncpus=2:mem=240gb:cluster=alfrid
#PBS -l walltime=48:00:00

module add python-3.6.2-gcc
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1

source /storage/brno6/home/wollf/learned-indexes/venv/bin/activate
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/python_runner_scripts/mocap

python3 mocap_performance_bayesian-gmm_regular_no-AE_3.py
