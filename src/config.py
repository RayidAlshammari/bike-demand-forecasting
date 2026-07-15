import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "raw" / "SeoulBikeData.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"

MODEL_PATH = MODEL_DIR / "model.onnx"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

NN_PARAMS = {
    'hidden_layer_sizes': (128, 64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'random_state': 42,
    'early_stopping': True
}