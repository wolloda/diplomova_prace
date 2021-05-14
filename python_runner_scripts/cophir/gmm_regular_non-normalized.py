import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs


benchmark = Benchmark(model = "GMM", dataset = DatasetDirs.COPHIR_1M, normalize = False)
benchmark.evaluate()
