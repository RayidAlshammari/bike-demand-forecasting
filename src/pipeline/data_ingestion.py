import pandas as pd
from src.config import RAW_DATA_PATH

def load_and_clean_data(file_path=RAW_DATA_PATH):
    print(f"[*] Loading raw data from {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='unicode_escape')
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    column_mapping = {
        'Rented Bike Count': 'Rented_Bike_Count',
        'Temperature(°C)': 'Temperature',
        'Humidity(%)': 'Humidity',
        'Wind speed (m/s)': 'Wind_Speed',
        'Visibility (10m)': 'Visibility',
        'Dew point temperature(°C)': 'Dew_Point_Temperature',
        'Solar Radiation (MJ/m2)': 'Solar_Radiation',
        'Rainfall(mm)': 'Rainfall',
        'Snowfall (cm)': 'Snowfall',
        'Functioning Day': 'Functioning_Day'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    print(f"[*] Data loaded successfully. Shape: {df.shape}")
    return df