import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "GMM", dataset = DatasetDirs.MOCAP, buckets = [8, 8, 8, 8, 8])
benchmark.evaluate([{'GMM': [{'comp': 8, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}]}, {'GMM': [{'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}]}, {'GMM': [{'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}, {'comp': 8, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}]}])
