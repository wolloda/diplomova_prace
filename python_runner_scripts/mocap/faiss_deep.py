import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.MOCAP, buckets = [10, 10, 10, 10, 10])
benchmark.evaluate()