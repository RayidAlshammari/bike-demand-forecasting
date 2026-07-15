import pandas as pd
import numpy as np

def engineer_features(df):
    print("[*] Starting feature engineering...")
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.day_name()
    
    df = pd.get_dummies(df, columns=['Seasons', 'Holiday', 'Functioning_Day'])
    
    conditions = [
        (df['Weekday'].isin(['Saturday', 'Sunday'])) | (df['Holiday_Holiday'] == 1),
        (~df['Weekday'].isin(['Saturday', 'Sunday'])) & (df['Holiday_Holiday'] == 0)
    ]
    df['Day_Type'] = np.select(conditions, ['Leisure', 'Work'], default='Work')
    df = pd.get_dummies(df, columns=['Day_Type'])
    
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda x: 1 if x in [8, 18, 19] else 0)
    df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
    
    cols_to_drop = [
        'Date', 'Weekday', 'Dew_Point_Temperature', 
        'Holiday_No Holiday', 'Functioning_Day_No', 
        'Hour', 'Month'
    ]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    print(f"[*] Feature engineering completed. Final shape: {df.shape}")
    return df