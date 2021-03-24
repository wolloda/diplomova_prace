from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "KMeans", dataset = DatasetDirs.PROFI_1M, descriptors = 2048)
benchmark.evaluate()
