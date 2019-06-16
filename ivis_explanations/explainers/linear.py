import numpy as np

from sklearn.linear_model import LinearRegression


class LinearExplainer(object):

    def __init__(self, model, data=None):
        self.model = model
        self.data = data

    def feature_importances_(self, X, **kwargs):
        if self.data is None:
            embeddings = self.model.transform(X)
        else:
            embeddings = self.data

        score = np.empty(shape=(X.shape[1],))
        for i in range(X.shape[1]):
            est = LinearRegression(**kwargs).fit(embeddings,
                                                 X[:, i])
            score[i] = est.score(embeddings, X[:, i])

        return score
