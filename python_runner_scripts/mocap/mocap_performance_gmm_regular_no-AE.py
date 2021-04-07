import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "GMM", dataset = DatasetDirs.MOCAP)
benchmark.evaluate([{'GMM': [{'comp': 5, 'cov_type': 'tied', 'max_iter': 2, 'init_params': 'kmeans'}, {'comp': 82, 'cov_type': 'tied', 'max_iter': 2, 'init_params': 'kmeans'}]}, {'GMM': [{'comp': 5, 'cov_type': 'tied', 'max_iter': 2, 'init_params': 'random'}, {'comp': 82, 'cov_type': 'tied', 'max_iter': 2, 'init_params': 'random'}]}, {'GMM': [{'comp': 5, 'cov_type': 'tied', 'max_iter': 5, 'init_params': 'kmeans'}, {'comp': 82, 'cov_type': 'tied', 'max_iter': 5, 'init_params': 'kmeans'}]}, {'GMM': [{'comp': 5, 'cov_type': 'tied', 'max_iter': 5, 'init_params': 'random'}, {'comp': 82, 'cov_type': 'tied', 'max_iter': 5, 'init_params': 'random'}]}, {'GMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 1, 'init_params': 'kmeans'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 1, 'init_params': 'kmeans'}]}, {'GMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 1, 'init_params': 'random'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 1, 'init_params': 'random'}]}, {'GMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'kmeans'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'kmeans'}]}, {'GMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 2, 'init_params': 'random'}]}, {'GMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'kmeans'}]}, {'GMM': [{'comp': 5, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}, {'comp': 82, 'cov_type': 'full', 'max_iter': 5, 'init_params': 'random'}]}])
