import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.COPHIR_1M)
benchmark.evaluate([{'Faiss': [{'n_clusters': 18, 'n_init': 15, 'LogReg': {'ep': 20}}, {'n_clusters': 94, 'n_init': 15, 'LogReg': {'ep': 20}}]}, {'Faiss': [{'n_clusters': 18, 'n_init': 20, 'LogReg': {'ep': 15}}, {'n_clusters': 94, 'n_init': 20, 'LogReg': {'ep': 15}}]}, {'Faiss': [{'n_clusters': 18, 'n_init': 20, 'LogReg': {'ep': 5}}, {'n_clusters': 94, 'n_init': 20, 'LogReg': {'ep': 5}}]}])
