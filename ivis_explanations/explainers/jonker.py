import lapjv
import numpy as np

from scipy.spatial.distance import cdist


class JVExplainer(object):

    def __init__(self, model,
                 random_state=None):
        self.model = model
        self.random_state = random_state

    def feature_importances_(self, X):
        if X.shape[1] > 2:
            embeddings = self.model.transform(X)
        else:
            embeddings = X

        embeddings -= embeddings.min(axis=0)
        embeddings /= embeddings.max(axis=0)

        dim = np.round(np.sqrt(embeddings.shape[0]))
        xv, yv = np.meshgrid(np.linspace(0, 1, dim),
                             np.linspace(0, 1, dim))

        grid = np.dstack((xv, yv)).reshape(-1, 2)

        cost_matrix = cdist(grid, embeddings, "sqeuclidean").astype(int)
        cost_matrix = cost_matrix * (10000000. / cost_matrix.max())

        row_index, col_index, _ = lapjv.lapjv(cost_matrix)

        grid_jv = grid[col_index]
        return grid_jv
