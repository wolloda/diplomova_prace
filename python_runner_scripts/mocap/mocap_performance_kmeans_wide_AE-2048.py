import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

benchmark = Benchmark(model = "KMeans", dataset = DatasetDirs.MOCAP, descriptors = 2048, buckets = [71, 71])
benchmark.evaluate()
