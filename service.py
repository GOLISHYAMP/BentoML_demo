import bentoml
import numpy as np
import os
from bentoml.io import NumpyNdarray

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    # Retrieve the latest version of the model from the BentoML Model Store
    bento_model = bentoml.sklearn.get("iris_clf:latest")

    def __init__(self):
        # Load the model
        self.model = bentoml.sklearn.load_model(self.bento_model.tag)

    @bentoml.api
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
    
    @bentoml.api
    def helloworld(self):
        return "All fine"
