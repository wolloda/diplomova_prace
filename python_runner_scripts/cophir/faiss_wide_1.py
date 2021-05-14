import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.COPHIR_1M, buckets = [100, 100])
benchmark.evaluate([{'Faiss': [{'n_clusters': 100, 'n_init': 5, 'LogReg': {'ep': 20}}, {'n_clusters': 100, 'n_init': 5, 'LogReg': {'ep': 20}}]}])
