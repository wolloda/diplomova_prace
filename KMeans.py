from BaseClassifier import BaseClassifier, logging
from sklearn.cluster import KMeans

class KMeansModel(BaseClassifier):
    def __init__(self):
        pass

    def train(self, kmeans_model, X, y, descriptor_values):
        """
        Trains a KMeans model.
        Expects model_dict to contain hyperparameters *comp* and *cov_type* (n. of components and covariance type)

        Parameters
        -------
        kmeans_model: Dict
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
        assert "n_clusters" in kmeans_model

        n_init = kmeans_model["n_init"] if "n_init" in kmeans_model else 10
        max_iter = kmeans_model["max_iter"] if "max_iter" in kmeans_model else 100

        logging.info(f'Training KMeans model with values of shape {X.shape}: n. of clusters={kmeans_model["n_clusters"]} | initializations={n_init} | max iterations={max_iter}')
        n_clusters = kmeans_model["n_clusters"]
        if X.shape[0] <= kmeans_model["n_clusters"]:
            previous_kmeans_clusters = kmeans_model["n_clusters"]
            n_clusters = X.shape[0]

            logging.warn(f"Reducing the number of components from {previous_kmeans_clusters} to {n_clusters} since the number of\
                           training samples ({X.shape[0]}) is less than {previous_kmeans_clusters}")
        root = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
        return super().train_unsupervised(root, X)

    def predict(self, query, model, encoder, value="value_l"):
        if model is not None:
            dist_distr = model.transform(query)[0]
            dist_distr = max(dist_distr) - dist_distr
            dist_distr += 1e-8
            dist_distr /= sum(dist_distr)
            return super().predict(dist_distr, encoder)
        else:
            return [1]

