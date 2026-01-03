from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize FastAPI app
app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# Define the input data model
class DiabetesData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str

# Load the trained model
try:
    # Load the trained RandomForest model
    # You need to save the model first from the notebook: joblib.dump(best_model_rf, 'diabetes_model.pkl')
    best_model_rf = joblib.load('DiabetesPredictionModel/diabetes_model.pkl')
except:
    # If model file doesn't exist, create a placeholder
    best_model_rf = None

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to Diabetes Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": best_model_rf is not None}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_diabetes(data: DiabetesData):
    """
    Predict diabetes based on patient health metrics
    """
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame({
            'Pregnancies': [data.Pregnancies],
            'Glucose': [data.Glucose],
            'BloodPressure': [data.BloodPressure],
            'SkinThickness': [data.SkinThickness],
            'Insulin': [data.Insulin],
            'BMI': [data.BMI],
            'DiabetesPedigreeFunction': [data.DiabetesPedigreeFunction],
            'Age': [data.Age]
        })
        
        # Make prediction
        prediction = best_model_rf.predict(input_df)[0]
        
        # Get prediction probability
        probabilities = best_model_rf.predict_proba(input_df)[0]
        probability = float(probabilities[int(prediction)])
        
        # Create message based on prediction
        if prediction == 1:
            message = "High risk of diabetes. Please consult a healthcare professional."
        else:
            message = "Low risk of diabetes. Maintain healthy lifestyle."
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            message=message
        )
    
    except Exception as e:
        return {
            "error": str(e),
            "message": "An error occurred during prediction"
        }

# Batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(data_list: list[DiabetesData]):
    """
    Predict diabetes for multiple patients at once
    """
    try:
        # Convert list of DiabetesData objects to DataFrame
        records = [d.dict() for d in data_list]
        input_df = pd.DataFrame(records)
        
        # Make predictions
        predictions = best_model_rf.predict(input_df)
        probabilities = best_model_rf.predict_proba(input_df)
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            prob = float(probabilities[i][int(pred)])
            message = "High risk" if pred == 1 else "Low risk"
            results.append({
                "patient_id": i,
                "prediction": int(pred),
                "probability": prob,
                "message": message
            })
        
        return {"results": results}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
