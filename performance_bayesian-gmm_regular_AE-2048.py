from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "BayesianGMM", dataset = DatasetDirs.PROFI_1M, descriptors = 2048)
benchmark.evaluate()
