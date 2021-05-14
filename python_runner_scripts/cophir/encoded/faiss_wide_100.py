import sys
sys.path.append("../../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.COPHIR_1M, buckets = [100, 100], descriptors = 100)
benchmark.evaluate()
