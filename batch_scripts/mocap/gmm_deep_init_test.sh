#!/bin/bash
#PBS -N mocap_performance_testing_gmm_deep_init_params
#PBS -l select=1:ncpus=2:mem=140gb:cluster=alfrid
#PBS -l walltime=16:00:00

module add python-3.6.2-gcc
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1

source /storage/brno6/home/wollf/learned-indexes/venv/bin/activate
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/python_runner_scripts/mocap

python3 mocap_performance_gmm_deep_init_test.py
