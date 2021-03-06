import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "BayesianGMM", dataset = DatasetDirs.MOCAP)
benchmark.evaluate([{'BayesianGMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 1, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_process'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 1, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_process'}]}, {'BayesianGMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_process'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_process'}]}, {'BayesianGMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_distribution'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_distribution'}]}])
