import joblib
import onnxruntime as rt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import SCALER_PATH, MODEL_PATH

scaler = None
onnx_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This is done once.(Startup)
    global scaler, onnx_session
    print("[*] Loading Scaler and ONNX Model...")
    try:
        scaler = joblib.load(SCALER_PATH)
        onnx_session = rt.InferenceSession(str(MODEL_PATH))
        print("[*] Model components loaded successfully.")
    except Exception as e:
        print(f"Error loading model components: {e}")
    yield
    print("[*] Shutting down...")

app = FastAPI(title="Seoul Bike Demand Forecaster API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Bike Demand Forecasting API is running. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionResponse)
def predict_demand(request: PredictionRequest):
    try:
        # 1. Convert incoming JSON request to a dictionary, then to a DataFrame (1 row)
        data = request.model_dump()
        df = pd.DataFrame([data])
        
        # 2. Re-apply the EXACT same feature engineering from training
        # Create Dummy variables manually based on our schema
        df['Seasons_Autumn'] = 1 if data['Season'] == 'Autumn' else 0
        df['Seasons_Spring'] = 1 if data['Season'] == 'Spring' else 0
        df['Seasons_Summer'] = 1 if data['Season'] == 'Summer' else 0
        df['Seasons_Winter'] = 1 if data['Season'] == 'Winter' else 0
        
        df['Holiday_Holiday'] = 1 if data['Holiday'] == 'Holiday' else 0
        df['Functioning_Day_Yes'] = 1 if data['Functioning_Day'] == 'Yes' else 0
        
        df['Day_Type_Leisure'] = 1 if data['Day_Type'] == 'Leisure' else 0
        df['Day_Type_Work'] = 1 if data['Day_Type'] == 'Work' else 0
        
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Is_Peak_Hour'] = 1 if df['Hour'][0] in [8, 18, 19] else 0
        df['Temp_Humidity_Interaction'] = df['Temperature'][0] * df['Humidity'][0]
        
        # Ensure correct column order exactly as the model expects
        expected_columns = [
            'Temperature', 'Humidity', 'Wind_Speed', 'Visibility', 'Solar_Radiation',
            'Rainfall', 'Snowfall', 'Seasons_Autumn', 'Seasons_Spring', 'Seasons_Summer',
            'Seasons_Winter', 'Holiday_Holiday', 'Functioning_Day_Yes', 'Day_Type_Leisure',
            'Day_Type_Work', 'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
            'Is_Peak_Hour', 'Temp_Humidity_Interaction'
        ]
        
        features_df = df[expected_columns]
        
        features_scaled = scaler.transform(features_df)
        
        # Predict using ONNX Runtime
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        prediction = onnx_session.run([output_name], {input_name: features_scaled.astype(np.float32)})[0]
        predicted_count = int(prediction[0][0])
        
        # Prevent negative predictions
        predicted_count = max(0, predicted_count)
        
        return {
            "expected_bike_demand": predicted_count,
            "message": "Prediction calculated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))