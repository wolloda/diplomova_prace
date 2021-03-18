from BaseClassifier import BaseClassifier, logging
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel(BaseClassifier):

    def __init__(self):
        pass

    def train(self, gmm_model, X, y, descriptor_values):
        """ Trains a Gaussian mixture model. Expects model_dict to contain hyperparameters *comp* and *cov_type* (n. of components and covariance type)
        
        Parameters
        -------
        gmm_model: Dict
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
        assert "comp" in gmm_model and "cov_type" in gmm_model

        max_iter = gmm_model["max_iter"] if "max_iter" in gmm_model else 100
        init_params = gmm_model["init_params"] if "init_params" in gmm_model else 'kmeans'

        logging.info(f'Training GMM model with values of shape {X.shape}: n. of clusters={gmm_model["comp"]} | covariance type={gmm_model["cov_type"]} | max iterations={max_iter} | initial parameters={init_params}')
        if X.shape[0] <= gmm_model["comp"]:
            previous_gmm_comp = gmm_model["comp"]
            gmm_model["comp"] = X.shape[0] // 2
            logging.warn(f"Reducing the number of components from {previous_gmm_comp} to {gmm_model['comp']} since the number of\
                           training samples ({X.shape[0]}) is less than {previous_gmm_comp}")
        root = GaussianMixture(n_components=gmm_model["comp"], covariance_type=gmm_model["cov_type"], max_iter=max_iter, init_params=init_params)
        return super().train(root, X, y=None, descriptor_values = descriptor_values)
    
    def predict(self, query, model, encoder, value="value_l"):
        prob_distr = model.predict_proba(query)[0]
        return super().predict(prob_distr, encoder)
