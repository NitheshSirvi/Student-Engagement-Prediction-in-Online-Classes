from lightgbm import LGBMClassifier
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("lightgbm_model")
        self.model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)