from xgboost import XGBClassifier
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("xgboost_model")
        self.model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)