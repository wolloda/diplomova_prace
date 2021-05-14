#!/bin/bash
#PBS -N faiss_regular.sh
#PBS -l select=1:ncpus=2:ngpus=1:mem=110gb:cluster=adan -q gpu
#PBS -l walltime=23:59:00

module add python-3.6.2-gcc
module add cuda-10.1
module add cudnn-7.6.4-cuda10.1

source /storage/brno6/home/wollf/learned-indexes/venv/bin/activate
cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/python_runner_scripts/cophir

python3  faiss_regular_2.py
