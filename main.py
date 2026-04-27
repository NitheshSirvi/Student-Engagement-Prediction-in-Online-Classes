import joblib
import os
from data.data_generator import generate_synthetic_data
from data.data_loader import load_data
from features.feature_engineering import prepare_data
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.neural_network import NeuralNetModel
from evaluation.metrics import evaluate_model
from utils.visualization import plot_confusion_matrix
from config.config import MODEL_SAVE_DIR

def main():
    # 1. Generate & Load Data
    generate_synthetic_data()
    df = load_data()
    
    if df is None:
        return

    # 2. Preprocess Data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    # Save the scaler for the API
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))

    # 3. Initialize Models
    models = [
        XGBoostModel(),
        LightGBMModel(),
        NeuralNetModel()
    ]

    # 4. Train, Evaluate, and Save
    for model in models:
        print(f"\nTraining {model.model_name}...")
        model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        evaluate_model(y_test, predictions, model.model_name)
        plot_confusion_matrix(y_test, predictions, model.model_name)
        
        model.save()
        
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()