from sklearn.neural_network import MLPClassifier
from .base_model import BaseModel

class NeuralNetModel(BaseModel):
    def __init__(self):
        super().__init__("neural_network_model")
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)