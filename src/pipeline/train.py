import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.config import NN_PARAMS, SCALER_PATH, MODEL_PATH, PROCESSED_DATA_PATH
from src.pipeline.data_ingestion import load_and_clean_data
from src.pipeline.features import engineer_features
from src.pipeline.evaluate import evaluate_model

def run_training_pipeline():
    print("=== Starting MLOps Training Pipeline ===")
    
    df = load_and_clean_data()
    df_processed = engineer_features(df)
    
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    
    X = df_processed.drop('Rented_Bike_Count', axis=1)
    y = df_processed['Rented_Bike_Count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"[*] Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    
    print("[*] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"[*] Scaler saved to {SCALER_PATH}")
    
    print(f"[*] Training Neural Network with params: {NN_PARAMS}")
    model = MLPRegressor(**NN_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    evaluate_model(model, X_test_scaled, y_test)
    
    print("[*] Converting model to ONNX format...")
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[*] ONNX Model saved to {MODEL_PATH}")
    
    print("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    run_training_pipeline()