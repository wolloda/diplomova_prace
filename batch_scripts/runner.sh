#!/bin/bash

cd /storage/brno6/home/wollf/learned-indexes/learned-indexes/batch_scripts

qsub performance_gmm_deep_AE-2048.sh >> my_jobs.txt
qsub performance_gmm_deep_AE-3277.sh >> my_jobs.txt
qsub performance_gmm_deep_no-AE.sh >> my_jobs.txt
qsub performance_gmm_wide_AE-2048.sh >> my_jobs.txt
qsub performance_gmm_wide_AE-3277.sh >> my_jobs.txt
qsub performance_gmm_wide_no-AE.sh >> my_jobs.txt
qsub performance_gmm_regular_AE-2048.sh >> my_jobs.txt
qsub performance_gmm_regular_AE-3277.sh >> my_jobs.txt
qsub performance_gmm_regular_no-AE.sh >> my_jobs.txt
qsub performance_bayesian-gmm_deep_AE-2048.sh >> my_jobs.txt
qsub performance_bayesian-gmm_deep_AE-3277.sh >> my_jobs.txt
qsub performance_bayesian-gmm_deep_no-AE.sh >> my_jobs.txt
qsub performance_bayesian-gmm_wide_AE-2048.sh >> my_jobs.txt
qsub performance_bayesian-gmm_wide_AE-3277.sh >> my_jobs.txt
qsub performance_bayesian-gmm_wide_no-AE.sh >> my_jobs.txt
qsub performance_bayesian-gmm_regular_AE-2048.sh >> my_jobs.txt
qsub performance_bayesian-gmm_regular_AE-3277.sh >> my_jobs.txt
qsub performance_bayesian-gmm_regular_no-AE.sh >> my_jobs.txt
qsub performance_kmeans_deep_AE-2048.sh >> my_jobs.txt
qsub performance_kmeans_deep_AE-3277.sh >> my_jobs.txt
qsub performance_kmeans_deep_no-AE.sh >> my_jobs.txt
qsub performance_kmeans_regular_AE-2048.sh >> my_jobs.txt
qsub performance_kmeans_regular_AE-3277.sh >> my_jobs.txt
qsub performance_kmeans_regular_no-AE.sh >> my_jobs.txt
qsub performance_kmeans_wide_AE-2048.sh >> my_jobs.txt
qsub performance_kmeans_wide_AE-3277.sh >> my_jobs.txt
qsub performance_kmeans_wide_no-AE.sh >> my_jobs.txt
qsub performance_faiss-logreg_deep_AE-2048.sh >> my_jobs.txt
qsub performance_faiss-logreg_deep_AE-3277.sh >> my_jobs.txt
qsub performance_faiss-logreg_deep_no-AE.sh >> my_jobs.txt
qsub performance_faiss-logreg_regular_AE-2048.sh >> my_jobs.txt
qsub performance_faiss-logreg_regular_AE-3277.sh >> my_jobs.txt
qsub performance_faiss-logreg_regular_no-AE.sh >> my_jobs.txt
qsub performance_faiss-logreg_wide_AE-2048.sh >> my_jobs.txt
qsub performance_faiss-logreg_wide_AE-3277.sh >> my_jobs.txt
qsub performance_faiss-logreg_wide_no-AE.sh >> my_jobs.txt
