from Benchmark import Benchmark
from enums import DatasetDirs

dataset = DatasetDirs.PROFI_1M

benchmark = Benchmark(
        model = "BayesianGMM",
        dataset = DatasetDirs.COPHIR_100k
        )

benchmark.evaluate()
