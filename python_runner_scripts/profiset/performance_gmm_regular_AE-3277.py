import sys
sys.path.append("..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "GMM", dataset = DatasetDirs.PROFI_1M, descriptors = 3277)
benchmark.evaluate()
