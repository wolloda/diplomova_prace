import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.MOCAP, buckets = [71, 71])
benchmark.evaluate([{'Faiss': [{'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 5}}, {'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 5}}]}, {'Faiss': [{'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 10}}, {'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 10}}]}, {'Faiss': [{'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 15}}, {'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 15}}]}, {'Faiss': [{'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 20}}, {'n_clusters': 71, 'n_init': 20, 'LogReg': {'ep': 20}}]}])
