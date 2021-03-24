from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.PROFI_1M, descriptors = 2048, buckets = [100, 100])
benchmark.evaluate()
