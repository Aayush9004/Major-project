import os
import joblib
import pandas as pd
from src.features import FEATURE_COLUMNS

def predict_candidate(features: dict, model_path: str = "models/random_forest.pkl") -> dict:
    """
    Predict candidate suitability using trained Random Forest model.
    
    Args:
        features (dict): Feature dictionary matching FEATURE_COLUMNS keys.
        model_path (str): Path to saved model file.
        
    Returns:
        dict: {"prediction": "SUITABLE" | "NOT SUITABLE", "probability": float}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train the model first by running `python main.py --train`."
        )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model file at {model_path}: {str(e)}")

    # Ensure feature vector matches FEATURE_COLUMNS order
    feature_values = [features.get(col, 0.0) for col in FEATURE_COLUMNS]
    df_features = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    try:
        pred_class = model.predict(df_features)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_features)[0]
            # Prob of class 1 (Suitable)
            prob_suitable = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            prob_suitable = 1.0 if pred_class == 1 else 0.0
    except Exception as e:
        raise ValueError(f"Error during model prediction: {str(e)}")

    prediction_str = "SUITABLE" if pred_class == 1 else "NOT SUITABLE"

    return {
        "prediction": prediction_str,
        "probability": round(prob_suitable, 4)
    }
