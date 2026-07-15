import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, model_name="Neural Network"):
    """
    Evaluates the model using RMSE and R2 metrics.
    """
    print(f"[*] Evaluating {model_name}...")
    
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"--- Evaluation Results ---")
    print(f"Model : {model_name}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"R2    : {r2:.4f}")
    print("--------------------------")
    
    return {"rmse": rmse, "r2": r2}