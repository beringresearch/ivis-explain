from sklearn.datasets import load_iris

from ivis import Ivis
from ivis_explanations import LinearExplainer

X = load_iris()['data']

ivis = Ivis(k=5, batch_size=28, model='maaten')
ivis.fit(X)

explainer = LinearExplainer(ivis)
explainer.feature_importances_(X)
