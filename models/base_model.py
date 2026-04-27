from abc import ABC, abstractmethod
import joblib
import os
from config.config import MODEL_SAVE_DIR

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def save(self):
        path = os.path.join(MODEL_SAVE_DIR, f"{self.model_name}.pkl")
        joblib.dump(self.model, path)
        print(f"Model saved: {path}")