import sys
sys.path.append("..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "BayesianGMM", dataset = DatasetDirs.PROFI_1M, descriptors = 3277, buckets = [100, 100])
benchmark.evaluate()
