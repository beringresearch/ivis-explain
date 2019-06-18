import lapjv
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.utils import shuffle


class JVExplainer(object):

    def __init__(self, model,
                 random_state=None):
        self.model = model

    def feature_importances_(self, X):
        if X.shape[0] > 10000:
            X = shuffle(X, random_state=self.random_state,
                        n_samples=10000)

        if X.shape[1] > 2:
            embeddings = self.model.transform(X)
        else:
            embeddings = X

        grid = np.stack((np.linspace(0, 1, embeddings.shape[0]),
                         np.linspace(0, 1, embeddings.shape[0])), axis=1)

        cost_matrix = cdist(grid, embeddings, "sqeuclidean").astype(np.float32)
        cost_matrix = cost_matrix * (embeddings.shape[0] / cost_matrix.max())

        row_asses, col_asses, _ = lapjv.lapjv(cost_matrix)

        grid_jv = grid[col_asses]
        return grid_jv
