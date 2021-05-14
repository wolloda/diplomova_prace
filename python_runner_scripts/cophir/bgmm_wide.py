import sys
sys.path.append("../..")
from Benchmark import Benchmark
from enums import DatasetDirs

buckets = [100, 100]
training_specs = []

covariance_types = ['spherical', 'tied', 'diag', 'full']
max_iter = [1, 2]
init_params = ['random']
weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']

for covariance_type in covariance_types:
    for iterations in max_iter:
        for init_param in init_params:
            for weight_concentration_prior_type in weight_concentration_prior_types:
                tmp = []
                for i in range(len(buckets)):
                    tmp.append({'comp': buckets[i], 'cov_type': covariance_type, 'max_iter': iterations, 'init_params': init_param, 'weight_concentration_prior_type': weight_concentration_prior_type})

                training_specs.append({"BayesianGMM": tmp})


benchmark = Benchmark(model = "BayesianGMM", dataset = DatasetDirs.COPHIR_1M, buckets = buckets)
benchmark.evaluate(training_specs)
