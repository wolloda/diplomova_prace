import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "GMM", dataset = DatasetDirs.COPHIR_1M, buckets = [5, 5])
benchmark.evaluate()
