def generate_specs(model, buckets):
    training_specs = []

    if model == "GMM":
        covariance_types = ['spherical', 'diag', 'tied', 'full']
        max_iter = [1, 2, 5]
        init_params = ['kmeans', 'random']

        for covariance_type in covariance_types:
            for iterations in max_iter:
                for init_param in init_params:
                    tmp = []
                    for i in range(len(buckets)):
                        tmp.append({'comp': buckets[i], 'cov_type': covariance_type, 'max_iter': iterations, 'init_params': init_param})

                    training_specs.append({model: tmp})

    elif model == "BayesianGMM":
        covariance_types = ['spherical', 'tied', 'diag', 'full']
        max_iter = [1, 2]
        init_params = ['kmeans', 'random']
        weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']

        for covariance_type in covariance_types:
            for iterations in max_iter:
                for init_param in init_params:
                    for weight_concentration_prior_type in weight_concentration_prior_types:
                        tmp = []
                        for i in range(len(buckets)):
                            tmp.append({'comp': buckets[i], 'cov_type': covariance_type, 'max_iter': iterations, 'init_params': init_param, 'weight_concentration_prior_type': weight_concentration_prior_type})

                        training_specs.append({model: tmp})

    elif model == "KMeans":
        max_iter = [5, 10, 25]
        n_init = [1, 5, 10]
        
        for iterations in max_iter:
            for initializations in n_init:
                tmp = []
                for i in range(len(buckets)):
                    tmp.append({'n_clusters': buckets[i], 'n_init': initializations, 'max_iter': iterations})

                training_specs.append({model: tmp})

    elif model == "Faiss":
        n_inits = [5, 10, 15, 20]
        log_reg_iterations = [5, 10, 15, 20]

        buckets = list(map(lambda x: int(x), buckets))

        for init in n_inits:
            for iteration in log_reg_iterations:
                tmp = []
                for i in range(len(buckets)):
                    tmp.append({'n_clusters': buckets[i], 'n_init': init, "LogReg": {"ep": iteration}})

                training_specs.append({model: tmp})

    return training_specs


print(generate_specs("BayesianGMM", [100, 100]))
