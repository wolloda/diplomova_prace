import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "Faiss", dataset = DatasetDirs.MOCAP, descriptors = 3277, buckets = [8, 8, 8, 8, 8])
benchmark.evaluate()
