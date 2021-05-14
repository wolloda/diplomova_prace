import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

buckets = [100, 100]
model = "GMM"

training_specs = []

covariance_types = ['spherical', 'diag', 'tied', 'full']
max_iter = [1, 2, 5]
init_params = ['random']

for covariance_type in covariance_types:
    for iterations in max_iter:
        for init_param in init_params:
            tmp = []
            for i in range(len(buckets)):
                tmp.append({'comp': buckets[i], 'cov_type': covariance_type, 'max_iter': iterations, 'init_params': init_param})

            training_specs.append({model: tmp})

benchmark = Benchmark(model = model, dataset = DatasetDirs.COPHIR_1M, buckets = buckets)
benchmark.evaluate(training_specs)
