from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "BayesianGMM", dataset = DatasetDirs.COPHIR_1M, buckets = [100, 100])
benchmark.evaluate()
