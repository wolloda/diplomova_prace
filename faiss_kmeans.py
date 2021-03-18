import faiss
from imports import np, pd

class FaissKMeans:
    def __init__(self, n_clusters=19, n_init=10, max_iter=300, gpu=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = gpu

    def fit(self, X):
        self.kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=self.n_clusters,
            niter=self.max_iter,
            nredo=self.n_init,
            gpu=self.gpu,
        )
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)

    def prepare(self, df):
        if len(df.shape) == 1:
            return np.ascontiguousarray([df.drop(["L1", "L2", "object_id"])])

        return np.ascontiguousarray(df.drop(["L1", "L2", "object_id"], axis=1))

