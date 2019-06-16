# ivis-explain

## Installation

```bash
git clone https://github.com/beringresearch/ivis-explain
cd ivis-explain
pip install --editable .
```

## Example

```python
from sklearn.datasets import load_iris

from ivis import Ivis
from ivis_explanations import LinearExplainer

ivis = Ivis(k=5, batch_size=28, model='maaten')
ivis.fit(X)

explainer = LinearExplainer(ivis)
explainer.feature_importances_(X)

>>> array([0.59248609, 0.65250957, 0.94732282, 0.94245944])
```


