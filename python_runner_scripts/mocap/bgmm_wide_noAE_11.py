import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "BayesianGMM", dataset = DatasetDirs.MOCAP, buckets = [71, 71])

benchmark.evaluate([{'BayesianGMM': [{'comp': 71, 'cov_type': 'tied', 'max_iter': 1, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_distribution'}, {'comp': 71, 'cov_type': 'tied', 'max_iter': 1, 'init_params': 'kmeans', 'weight_concentration_prior_type': 'dirichlet_distribution'}]}])
