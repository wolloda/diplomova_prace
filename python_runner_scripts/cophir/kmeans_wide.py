import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "KMeans", dataset = DatasetDirs.COPHIR_1M, buckets = [100, 100])
benchmark.evaluate()
