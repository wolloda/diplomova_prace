import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.COPHIR_1M, buckets = [100, 100])
benchmark.evaluate([{'Faiss': [{'n_clusters': 100, 'n_init': 10, 'LogReg': {'ep': 15}}, {'n_clusters': 100, 'n_init': 10, 'LogReg': {'ep': 15}}]}])
