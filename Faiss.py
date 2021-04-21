from sklearn.linear_model import LogisticRegression
from BaseClassifier import BaseClassifier, logging
from utils import get_class_weights_dict
import faiss

from imports import np, pd
from utils import get_knn_objects, get_sample_1k_objects

from LogisticRegression import custom_softmax, BarebonesLogisticRegression
from faiss_kmeans import FaissKMeans

def unify_types(df):
    df_modified = df.drop(["object_id"], axis = 1).astype(np.float32)
    df_modified["object_id"] = df["object_id"].astype(np.int64)

    cols = df_modified.columns.tolist()
    cols = cols[0:2] + cols[-1:] + cols[2:-1]
    df_modified = df_modified[cols]

    return df_modified

class Faiss(BaseClassifier):
    def __init__(self):
        pass

    def train(self, model, X, y, descriptor_values):
        """
        Trains a FaissKmeans and a Logistic regression model.
        Expects model_dict to contain hyperparameter *ep* (number of epochs)

        Parameters
        -------
        lr_model: Dict
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

        assert "n_clusters" in model

        n_init = model["n_init"] if "n_init" in model else 10
        max_iter = model["max_iter"] if "max_iter" in model else 300
        gpu = model["gpu"] if "gpu" in model else True

        #df = unify_types(X)
        #df_data = df.drop(["L1", "L2", "object_id"], axis=1).values
        if X.shape[0] != 1:
            df_data = X
            n = df_data.shape[0]
            dimension = df_data.shape[1]

            db_vectors = np.ascontiguousarray(df_data)

            res = faiss.StandardGpuResources() # declaring a GPU resource, using all the available GPUs
            clusters = model["n_clusters"]
            if X.shape[0] < model["n_clusters"]:
                clusters = X.shape[0]
            fkm = FaissKMeans(n_clusters = clusters, n_init = n_init, max_iter = max_iter, gpu = gpu)
            fkm.fit(db_vectors)

            predictions = fkm.predict(db_vectors)[1]
        else:
            predictions = np.array([0])

        if "LogReg" in model:
            assert "ep" in model["LogReg"]

            logging.info(f'Training LogReg model with values of shape {X.shape}: epochs={model["LogReg"]["ep"]}')

            #d_class_weights = get_class_weights_dict(predictions)
            node = BarebonesLogisticRegression(max_iter = model["LogReg"]["ep"])
            return super().train(node, X, predictions.ravel(), descriptor_values)
        else:
            logging.warn(f"Did not recognize any known classifier from {model_dict}, exiting.")
            return None

    def predict(self, query, model, encoder):
        if model is not None:
            prob_distr = model.predict_proba_single(query)
        else:
            prob_distr = [1]
        return super().predict(prob_distr, encoder)

