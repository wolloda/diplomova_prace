from BaseClassifier import BaseClassifier, logging
from sklearn.mixture import BayesianGaussianMixture

class BayesianGaussianMixtureModel(BaseClassifier):

    def __init__(self):
        pass

    def train(self, bayesian_gmm, X, y, descriptor_values):
        """
        Trains a Bayesian Gaussian mixture model.
        Expects model_dict to contain hyperparameters *comp* and *cov_type* (n. of components and covariance type)
        
        Parameters
        -------
        bayesian_gmm: Dict
            Dictionary of model specification
        X: Numpy array
            Training values
        y: Numpy array
            Training labels 
        
        Returns
        -------
        predictions: Numpy array
            Array of model predictions
        
        encoder: LabelEncoder
            Mapping of actual labels to labels used in training
        """
        assert "comp" in bayesian_gmm and "cov_type" in bayesian_gmm


        # TODO: this is a hacky version-dependent solution (defaults may vary between scikit versions)
        max_iter = bayesian_gmm["max_iter"] if "max_iter" in bayesian_gmm else 100
        init_params = bayesian_gmm["init_params"] if "init_params" in bayesian_gmm else 'kmeans'
        weight_concentration_prior_type = bayesian_gmm["weight_concentration_prior_type"] if "init_params" in bayesian_gmm else 'dirichlet_process'

        logging.info(f'Training BayesianGMM model with values of shape {X.shape}: n. of clusters={bayesian_gmm["comp"]} | covariance type={bayesian_gmm["cov_type"]} | max_iter={max_iter} | init_params={init_params} | weight_concentration_prior_type={weight_concentration_prior_type}')
        if X.shape[0] <= bayesian_gmm["comp"]:
            previous_bayesian_gmm_comp = bayesian_gmm["comp"]
            bayesian_gmm["comp"] = X.shape[0] // 2
            logging.warn(f"Reducing the number of components from {previous_bayesian_gmm_comp} to {bayesian_gmm['comp']} since the number of\
                           training samples ({X.shape[0]}) is less than {previous_bayesian_gmm_comp}")
        root = BayesianGaussianMixture(n_components=bayesian_gmm["comp"], covariance_type=bayesian_gmm["cov_type"], max_iter=max_iter, init_params=init_params, weight_concentration_prior_type=weight_concentration_prior_type)
        return super().train(root, X, y=None, descriptor_values=descriptor_values)
    
    def predict(self, query, model, encoder, value="value_l"):
        prob_distr = model.predict_proba(query)[0]
        return super().predict(prob_distr, encoder)

