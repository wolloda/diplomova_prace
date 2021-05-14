import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.COPHIR_1M)
benchmark.evaluate([{'Faiss': [{'n_clusters': 18, 'n_init': 20, 'LogReg': {'ep': 10}}, {'n_clusters': 94, 'n_init': 20, 'LogReg': {'ep': 10}}]}, {'Faiss': [{'n_clusters': 18, 'n_init': 20, 'LogReg': {'ep': 20}}, {'n_clusters': 94, 'n_init': 20, 'LogReg': {'ep': 20}}]}])
