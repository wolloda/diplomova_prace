from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "KMeans", dataset = DatasetDirs.PROFI_1M, buckets = [10, 10, 10, 10, 10])
benchmark.evaluate()
